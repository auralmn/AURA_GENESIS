import { io, Socket } from 'socket.io-client';

class AURAWebSocketClient {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  connect(serverUrl: string = 'http://localhost:3001') {
    this.socket = io(serverUrl, {
      transports: ['websocket'],
      autoConnect: true,
    });

    this.socket.on('connect', () => {
      console.log('🔌 Connected to AURA WebSocket server');
      this.reconnectAttempts = 0;
    });

    this.socket.on('disconnect', () => {
      console.log('🔌 Disconnected from AURA WebSocket server');
    });

    this.socket.on('connect_error', (error) => {
      console.error('🔌 Connection error:', error);
      this.handleReconnect();
    });

    return this.socket;
  }

  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`🔄 Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      
      setTimeout(() => {
        this.socket?.connect();
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('❌ Max reconnection attempts reached');
    }
  }

  subscribeToHealthUpdates(callback: (data: any) => void) {
    if (this.socket) {
      this.socket.on('health_snapshot', callback);
    }
  }

  subscribeToFiringEvents(callback: (data: any) => void) {
    if (this.socket) {
      this.socket.on('firing_event', callback);
    }
  }

  subscribeToAlerts(callback: (data: any) => void) {
    if (this.socket) {
      this.socket.on('alert', callback);
    }
  }

  subscribeToSystemStatus(callback: (data: any) => void) {
    if (this.socket) {
      this.socket.on('system_status', callback);
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  isConnected(): boolean {
    return this.socket?.connected || false;
  }
}

export const auraWebSocket = new AURAWebSocketClient();
