/**
 * Audio Service
 *
 * Handles audio recording, processing, and playback
 */

import websocketService, { WebSocketService, MessageType } from './websocket';

// Audio configuration
interface AudioConfig {
  sampleRate: number;
  channelCount: number;
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
  bufferSize: number;
}

// Default audio configuration
const DEFAULT_CONFIG: AudioConfig = {
  sampleRate: 44100, // Match microphone's native sample rate
  channelCount: 1, // Mono
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
  bufferSize: 4096
};

// Audio service state
export enum AudioState {
  INACTIVE = 'inactive',
  RECORDING = 'recording',
  PLAYING = 'playing',
  SPEAKING = 'speaking',     // Playing TTS content specifically
  INTERRUPTED = 'interrupted'
}

// Audio service events
export enum AudioEvent {
  RECORDING_START = 'recording_start',
  RECORDING_STOP = 'recording_stop',
  RECORDING_DATA = 'recording_data',
  PLAYBACK_START = 'playback_start',
  PLAYBACK_STOP = 'playback_stop',
  PLAYBACK_END = 'playback_end',
  AUDIO_ERROR = 'audio_error',
  AUDIO_STATE_CHANGE = 'audio_state_change'
}

// Event listener interface
type AudioEventListener = (data: any) => void;

/**
 * Audio Service class
 */
export class AudioService {
  private config: AudioConfig;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private mediaStreamSource: MediaStreamAudioSourceNode | null = null;
  private scriptProcessor: ScriptProcessorNode | null = null;
  private recordingIntervalId: number | null = null;
  private recordingInterval: number = 100; // ms
  private audioBuffer: Float32Array[] = [];
  private audioState: AudioState = AudioState.INACTIVE;
  private eventListeners: Map<AudioEvent, AudioEventListener[]> = new Map();
  private audioQueue: AudioBuffer[] = [];
  private isPlaying: boolean = false;
  private isSpeaking: boolean = false; // Distinct from isPlaying to track TTS specifically
  private isMuted: boolean = false; // Track microphone mute state
  private currentSource: AudioBufferSourceNode | null = null;
  
  // State tracking (for UI coordination)
  private isProcessing: boolean = false;
  private isGreeting: boolean = false;
  private isVisionProcessing: boolean = false;
  
  // Voice detection parameters
  private isVoiceDetected: boolean = false;
  private voiceThreshold: number = 0.01; // Adjust based on testing
  private silenceTimeout: number = 1000; // ms to keep recording after voice drops below threshold
  private lastVoiceTime: number = 0;
  private minRecordingLength: number = 1000; // Minimum ms of audio to send

  constructor(config: Partial<AudioConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Set processing state from UI
   */
  public setProcessingState(isProcessing: boolean): void {
    this.isProcessing = isProcessing;
    console.log(`Processing state set to: ${isProcessing}`);
  }
  
  /**
   * Set greeting state from UI
   * This prevents interrupts during the initial greeting
   */
  public setGreetingState(isGreeting: boolean): void {
    this.isGreeting = isGreeting;
    console.log(`Greeting state set to: ${isGreeting}`);
  }

  /**
   * Initialize the audio context
   */
  private async initAudioContext(): Promise<void> {
    // If context is null, create a new one
    if (!this.audioContext) {
      console.log('Creating new AudioContext');
      try {
        this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
          sampleRate: this.config.sampleRate
        });
      } catch (error) {
        console.error('Failed to create AudioContext', error);
        this.dispatchEvent(AudioEvent.AUDIO_ERROR, { error });
        throw error;
      }
    }
    
    // Always make sure context is running
    if (this.audioContext.state === 'suspended') {
      console.log('Resuming suspended AudioContext');
      try {
        await this.audioContext.resume();
      } catch (error) {
        console.error('Failed to resume AudioContext', error);
        // If resume fails, try creating a new context
        this.audioContext = null;
        return this.initAudioContext();
      }
    } else if (this.audioContext.state === 'closed') {
      console.log('AudioContext was closed, creating new one');
      this.audioContext = null;
      return this.initAudioContext();
    }
    
    console.log(`AudioContext initialized, state: ${this.audioContext.state}`);
  }

  /**
   * Start recording audio
   */
  public async startRecording(): Promise<void> {
    if (this.audioState === AudioState.RECORDING) {
      console.log('Already recording');
      return;
    }

    try {
      await this.initAudioContext();
      
      // Request microphone access
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: this.config.sampleRate,
          channelCount: this.config.channelCount,
          echoCancellation: this.config.echoCancellation,
          noiseSuppression: this.config.noiseSuppression,
          autoGainControl: this.config.autoGainControl
        }
      });
      
      // Apply mute state if already set
      if (this.isMuted && this.mediaStream) {
        this.mediaStream.getAudioTracks().forEach(track => {
          track.enabled = !this.isMuted;
        });
      }
      
      // Create media stream source
      if (this.audioContext) {
        this.mediaStreamSource = this.audioContext.createMediaStreamSource(this.mediaStream);
        
        // Create script processor for recording
        this.scriptProcessor = this.audioContext.createScriptProcessor(
          this.config.bufferSize,
          this.config.channelCount,
          this.config.channelCount
        );
        
        // Connect nodes
        this.mediaStreamSource.connect(this.scriptProcessor);
        this.scriptProcessor.connect(this.audioContext.destination);
        
        // Handle audio processing
        this.scriptProcessor.onaudioprocess = this.handleAudioProcess.bind(this);
        
        // Clear previous buffer
        this.audioBuffer = [];
        
        // Set state
        this.audioState = AudioState.RECORDING;
        
        // Reset voice detection state
        this.isVoiceDetected = false;
        this.lastVoiceTime = 0;
        
        // Log voice detection threshold
        console.log(`Voice detection enabled with threshold: ${this.voiceThreshold}`);
        
        // Dispatch event
        this.dispatchEvent(AudioEvent.RECORDING_START, {});
        
        console.log('Recording started');
      }
    } catch (error) {
      console.error('Error starting recording:', error);
      this.dispatchEvent(AudioEvent.AUDIO_ERROR, { error });
      this.stopRecording();
      throw error;
    }
  }

  /**
   * Stop recording audio
   */
  public stopRecording(): void {
    if (this.audioState !== AudioState.RECORDING) {
      return;
    }

    // Stop sending chunks
    if (this.recordingIntervalId !== null) {
      clearInterval(this.recordingIntervalId);
      this.recordingIntervalId = null;
    }

    // Stop and clean up recorder
    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor = null;
    }

    if (this.mediaStreamSource) {
      this.mediaStreamSource.disconnect();
      this.mediaStreamSource = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    // Send any remaining audio data
    this.sendAudioChunk();

    // Reset state
    this.audioState = AudioState.INACTIVE;
    this.audioBuffer = [];

    // Dispatch event
    this.dispatchEvent(AudioEvent.RECORDING_STOP, {});
    
    console.log('Recording stopped');
  }

  /**
   * Calculate RMS (Root Mean Square) energy of an audio buffer
   */
  private calculateRMSEnergy(buffer: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) {
      sum += buffer[i] * buffer[i]; // Square each sample
    }
    const rms = Math.sqrt(sum / buffer.length); // RMS = square root of average
    return rms;
  }

  /**
   * Handle audio processing
   */
  private handleAudioProcess(event: AudioProcessingEvent): void {
    const inputBuffer = event.inputBuffer;
    const inputData = inputBuffer.getChannelData(0);
    
    // Create a copy of the buffer
    const bufferCopy = new Float32Array(inputData.length);
    bufferCopy.set(inputData);
    
    // Calculate RMS energy
    const energy = this.calculateRMSEnergy(bufferCopy);
    
    // Check if energy is above threshold (voice detected)
    if (energy > this.voiceThreshold) {
      // Check if in a protected state - if so, ignore voice detection entirely
      if (this.isProcessing || this.isVisionProcessing || this.isGreeting) {
        let state = "processing";
        if (this.isVisionProcessing) state = "vision_processing";
        if (this.isGreeting) state = "greeting";
        
        console.log(`Voice detected during ${state} (energy: ${energy.toFixed(4)}), ignoring`);
        // Skip further processing - don't even update isVoiceDetected
        
        // Still dispatch event for visualization, but mark isVoice as false
        this.dispatchEvent(AudioEvent.RECORDING_DATA, { 
          buffer: bufferCopy,
          energy: energy,
          isVoice: false // Force false during processing or greeting
        });
        
        return;
      }
      
      if (!this.isVoiceDetected) {
        console.log('Voice detected, energy:', energy);
        this.isVoiceDetected = true;
        
      // Check if we're currently playing TTS audio
      // If so, interrupt it immediately - BUT NOT during greeting
      // Also explicitly check audioState to catch any edge cases
      if ((this.isSpeaking || this.audioState === AudioState.SPEAKING) && !this.isGreeting) {
        console.log('User started speaking while assistant was speaking - interrupting playback',
                   `isSpeaking=${this.isSpeaking}, audioState=${this.audioState}, isGreeting=${this.isGreeting}`);
        // Stop playback locally
        this.stopPlayback();
        // Send interrupt signal to server
        websocketService.interrupt();
        // Dispatch an event so UI can update
        this.dispatchEvent(AudioEvent.PLAYBACK_STOP, {
          interrupted: true,
          reason: 'user_interrupt'
        });
      } else if (this.isGreeting) {
        console.log('Voice detected during greeting - suppressing interrupt');
      }
      }
      this.lastVoiceTime = Date.now();
    }
    
    // If in a protected state, never accumulate audio buffer
    if (this.isProcessing || this.isVisionProcessing || this.isGreeting) {
      // Dispatch event for visualization only
      this.dispatchEvent(AudioEvent.RECORDING_DATA, { 
        buffer: bufferCopy,
        energy: energy,
        isVoice: false // Force false during processing
      });
      return;
    }
    
    // Add to buffer if voice is detected or we're in the silence timeout period
    if (this.isVoiceDetected) {
      this.audioBuffer.push(bufferCopy);
      
      // Check if we've exceeded silence timeout
      const timeSinceVoice = Date.now() - this.lastVoiceTime;
      if (energy <= this.voiceThreshold && timeSinceVoice > this.silenceTimeout) {
        console.log('Voice ended, silence timeout exceeded');
        this.isVoiceDetected = false;
        
        // Send accumulated audio
        this.sendAudioChunk();
      }
    }
    
    // Dispatch event
    this.dispatchEvent(AudioEvent.RECORDING_DATA, { 
      buffer: bufferCopy,
      energy: energy,
      isVoice: this.isVoiceDetected
    });
  }

  /**
   * Convert Float32Array audio data to WAV format
   */
  private float32ToWav(buffer: Float32Array, sampleRate: number): ArrayBuffer {
    // Create buffer with WAV header
    const numChannels = 1; // Mono
    const bytesPerSample = 2; // 16-bit PCM
    const dataSize = buffer.length * bytesPerSample;
    const headerSize = 44; // Standard WAV header size
    const totalSize = headerSize + dataSize;
    
    // Create the WAV buffer
    const wavBuffer = new ArrayBuffer(totalSize);
    const wavView = new DataView(wavBuffer);
    
    // Write WAV header
    // "RIFF" chunk descriptor
    this.writeString(wavView, 0, 'RIFF');
    wavView.setUint32(4, totalSize - 8, true); // File size - 8
    this.writeString(wavView, 8, 'WAVE');
    
    // "fmt " sub-chunk
    this.writeString(wavView, 12, 'fmt ');
    wavView.setUint32(16, 16, true); // Sub-chunk size (16 for PCM)
    wavView.setUint16(20, 1, true); // Audio format (1 for PCM)
    wavView.setUint16(22, numChannels, true); // Number of channels
    wavView.setUint32(24, sampleRate, true); // Sample rate
    wavView.setUint32(28, sampleRate * numChannels * bytesPerSample, true); // Byte rate
    wavView.setUint16(32, numChannels * bytesPerSample, true); // Block align
    wavView.setUint16(34, bytesPerSample * 8, true); // Bits per sample
    
    // "data" sub-chunk
    this.writeString(wavView, 36, 'data');
    wavView.setUint32(40, dataSize, true); // Sub-chunk size
    
    // Write audio data
    // Convert from Float32 [-1.0,1.0] to Int16 [-32768,32767]
    const offset = 44;
    for (let i = 0; i < buffer.length; i++) {
      // Clamp the value to [-1.0, 1.0]
      const sample = Math.max(-1.0, Math.min(1.0, buffer[i]));
      // Convert to Int16
      const val = sample < 0 ? sample * 32768 : sample * 32767;
      wavView.setInt16(offset + i * bytesPerSample, val, true);
    }
    
    return wavBuffer;
  }
  
  /**
   * Helper function to write a string to a DataView
   */
  private writeString(view: DataView, offset: number, string: string): void {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  /**
   * Send accumulated audio chunk to WebSocket
   */
  private sendAudioChunk(): void {
    if (this.audioBuffer.length === 0) {
      return;
    }
    
    // Don't send audio if we're in processing state
    if (this.isProcessing) {
      console.log('Processing state active, discarding audio chunk');
      this.audioBuffer = [];
      return;
    }

    // Calculate total length
    const totalLength = this.audioBuffer.reduce((acc, buffer) => acc + buffer.length, 0);
    
    // Check if we have enough audio to send (avoid sending tiny fragments)
    const audioLengthMs = (totalLength / this.config.sampleRate) * 1000;
    if (!this.isVoiceDetected && audioLengthMs < this.minRecordingLength) {
      console.log(`Audio too short (${audioLengthMs.toFixed(0)}ms), discarding`);
      this.audioBuffer = [];
      return;
    }
    
    // Create combined buffer
    const combinedBuffer = new Float32Array(totalLength);
    
    // Copy data
    let offset = 0;
    for (const buffer of this.audioBuffer) {
      combinedBuffer.set(buffer, offset);
      offset += buffer.length;
    }
    
    console.log(`Sending audio chunk: ${audioLengthMs.toFixed(0)}ms`);
    
    // Convert to WAV format
    const wavBuffer = this.float32ToWav(combinedBuffer, this.config.sampleRate);
    
    // Send to WebSocket
    websocketService.sendAudio(wavBuffer);
    
    // Clear buffer
    this.audioBuffer = [];
  }

  /**
   * Play audio from base64-encoded data
   * 
   * The backend now sends complete audio files instead of chunks,
   * so we just need to decode and play the entire file at once.
   * 
   * This method is specifically for playing TTS content and will
   * set the state to SPEAKING rather than just PLAYING.
   */
  public async playAudioChunk(base64AudioChunk: string, format: string = 'wav'): Promise<void> {
    try {
      await this.initAudioContext();
      
      if (!this.audioContext) {
        throw new Error('AudioContext not initialized');
      }
      
      // Convert base64 to ArrayBuffer
      const audioData = WebSocketService.base64ToArrayBuffer(base64AudioChunk);
      
      console.log(`Received complete audio file (${audioData.byteLength} bytes)`);
      
      // Decode the audio data
      try {
        const audioBuffer = await this.audioContext.decodeAudioData(audioData);
        
        // Add to queue (instead of immediate playback)
        this.audioQueue.push(audioBuffer);
        
        // Start playback if not already playing
        if (!this.isPlaying) {
          this.playNextChunk();
        } else {
          console.log(`Added audio buffer to queue: duration=${audioBuffer.duration.toFixed(2)}s`);
        }
        
      } catch (error) {
        console.error('Error decoding audio data:', error);
        this.dispatchEvent(AudioEvent.AUDIO_ERROR, { error });
      }
    } catch (error) {
      console.error('Error queueing audio chunk:', error);
      this.dispatchEvent(AudioEvent.AUDIO_ERROR, { error });
    }
  }
  
  /**
   * Play next audio chunk from the queue
   */
  private playNextChunk(): void {
    console.log(`>> playNextChunk called. Queue length: ${this.audioQueue.length}, isPlaying: ${this.isPlaying}, isSpeaking: ${this.isSpeaking}`);
    
    if (this.audioQueue.length === 0) {
      this.isPlaying = false;
      this.isSpeaking = false;
      this.audioState = AudioState.INACTIVE;
      this.dispatchEvent(AudioEvent.PLAYBACK_END, {
        previousState: AudioState.SPEAKING
      });
      console.log('Audio queue empty, playback complete');
      return;
    }
    
    if (!this.audioContext) return;
    
    const buffer = this.audioQueue.shift();
    if (!buffer) return;
    
    // Set playback state - only dispatch PLAYBACK_START on the first buffer
    const wasPlaying = this.isPlaying;
    this.isPlaying = true;
    this.isSpeaking = true;
    this.audioState = AudioState.SPEAKING;
    
    // Create source node
    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);
    
    // Handle when this chunk ends
    source.onended = () => {
      console.log(`Buffer playback ended. Queue length: ${this.audioQueue.length}`);
      // If there are more chunks, play them
      if (this.audioQueue.length > 0) {
        this.playNextChunk();
      } else {
        // No more chunks, end playback
        this.isPlaying = false;
        this.isSpeaking = false;
        this.audioState = AudioState.INACTIVE;
        this.currentSource = null;
        this.dispatchEvent(AudioEvent.PLAYBACK_END, {
          previousState: AudioState.SPEAKING
        });
        console.log('Last audio chunk complete, playback ended');
      }
    };
    
    // Keep track of current source for stopping
    this.currentSource = source;
    
    // Start playback with a small delay
    source.start(this.audioContext.currentTime + 0.05);
    
    console.log(`Playing audio buffer: duration=${buffer.duration.toFixed(2)}s, queue remaining: ${this.audioQueue.length}`);
    
    // Dispatch playback start event only if we weren't already playing
    if (!wasPlaying) {
      console.log('First chunk in sequence - dispatching PLAYBACK_START event');
      this.dispatchEvent(AudioEvent.PLAYBACK_START, {});
    }
  }
  
  /**
   * Check if audio is currently playing speech
   */
  public isCurrentlySpeaking(): boolean {
    return this.isSpeaking;
  }
  
  /**
   * Get the length of the audio queue
   */
  public getAudioQueueLength(): number {
    return this.audioQueue.length;
  }
  
  /**
   * Check if microphone input is muted
   */
  public isMicrophoneMuted(): boolean {
    return this.isMuted;
  }
  
  /**
   * Toggle microphone mute state
   * Returns the new mute state
   */
  public toggleMicrophoneMute(): boolean {
    this.isMuted = !this.isMuted;
    
    // Apply mute state to active audio tracks
    if (this.mediaStream) {
      this.mediaStream.getAudioTracks().forEach(track => {
        track.enabled = !this.isMuted;
      });
      console.log(`Microphone ${this.isMuted ? 'muted' : 'unmuted'}`);
    } else {
      console.log('No active microphone to mute/unmute');
    }
    
    // Dispatch event
    this.dispatchEvent(AudioEvent.AUDIO_STATE_CHANGE, {
      type: 'mute_change',
      isMuted: this.isMuted
    });
    
    return this.isMuted;
  }

  /**
   * Stop audio playback
   */
  public stopPlayback(): void {
    if (!this.currentSource) {
      return;
    }
    
    // Store previous state for the event
    const previousState = this.audioState;
    
    try {
      this.currentSource.stop();
      this.currentSource = null;
    } catch (error) {
      console.error('Error stopping playback:', error);
    }
    
    // Clear the queue
    this.audioQueue = [];
    
    // Set state to INTERRUPTED if we were SPEAKING
    if (previousState === AudioState.SPEAKING) {
      this.audioState = AudioState.INTERRUPTED;
    } else {
      this.audioState = AudioState.INACTIVE;
    }
    
    this.isPlaying = false;
    this.isSpeaking = false;
    
    // Dispatch event with previous state info
    this.dispatchEvent(AudioEvent.PLAYBACK_STOP, { 
      interrupted: previousState === AudioState.SPEAKING,
      previousState: previousState
    });
    
    console.log('Playback stopped');
  }

  /**
   * Fully release all hardware access
   * This is more aggressive than just stopRecording() as it also:
   * - Forces all media tracks to stop
   * - Suspends the audio context
   * - Nullifies all resources
   * 
   * Use this when completely ending a call to ensure microphone
   * permissions are fully released at the hardware level.
   */
  public releaseHardware(): void {
    console.log('Releasing all hardware access...');
    
    // First stop any active recording/playback
    this.stopRecording();
    this.stopPlayback();
    
    // Force-stop and disable all tracks to release hardware
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => {
        track.stop();
        track.enabled = false;
      });
      this.mediaStream = null;
    }
    
    // Ensure script processor is disconnected
    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor = null;
    }
    
    // Ensure media stream source is disconnected
    if (this.mediaStreamSource) {
      this.mediaStreamSource.disconnect();
      this.mediaStreamSource = null;
    }
    
    // Suspend the audio context if it's running
    if (this.audioContext?.state === 'running') {
      this.audioContext.suspend().catch(err => {
        console.error('Error suspending audio context:', err);
      });
    }
    
    // Reset all state
    this.audioState = AudioState.INACTIVE;
    this.isVoiceDetected = false;
    this.audioBuffer = [];
    this.isPlaying = false;
    this.isSpeaking = false;
    
    console.log('All hardware access released');
  }

  /**
   * Get current audio state
   */
  public getAudioState(): AudioState {
    return this.audioState;
  }

  /**
   * Add event listener
   */
  public addEventListener(event: AudioEvent, callback: AudioEventListener): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    
    this.eventListeners.get(event)?.push(callback);
  }

  /**
   * Remove event listener
   */
  public removeEventListener(event: AudioEvent, callback: AudioEventListener): void {
    if (!this.eventListeners.has(event)) {
      return;
    }
    
    const listeners = this.eventListeners.get(event) || [];
    this.eventListeners.set(
      event,
      listeners.filter(listener => listener !== callback)
    );
  }

  /**
   * Dispatch event
   */
  private dispatchEvent(event: AudioEvent, data: any): void {
    if (!this.eventListeners.has(event)) {
      return;
    }
    
    const listeners = this.eventListeners.get(event) || [];
    listeners.forEach(listener => {
      try {
        listener(data);
      } catch (error) {
        console.error(`Error in ${event} listener:`, error);
      }
    });
  }
}

// Create singleton instance
const audioService = new AudioService();
export default audioService;
