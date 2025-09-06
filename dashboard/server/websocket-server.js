const { Server } = require('socket.io');
const http = require('http');
const { spawn } = require('child_process');
const path = require('path');

class AURAWebSocketServer {
  constructor(port = 3001) {
    this.port = port;
    this.server = http.createServer();
    this.io = new Server(this.server, {
      cors: {
        origin: "http://localhost:3000",
        methods: ["GET", "POST"]
      }
    });
    this.auraProcess = null;
    this.isConnected = false;
    this.setupSocketHandlers();
  }

  setupSocketHandlers() {
    this.io.on('connection', (socket) => {
      console.log('🔌 Client connected:', socket.id);
      
      socket.on('disconnect', () => {
        console.log('🔌 Client disconnected:', socket.id);
      });

      // Send initial connection status
      socket.emit('system_status', {
        type: 'connection',
        status: 'connected',
        timestamp: Date.now()
      });
    });
  }

  async startAURAMonitor() {
    console.log('🚀 Starting AURA health monitor...');
    
    // Path to the AURA health monitor script
    const auraScriptPath = path.join(__dirname, '../../aura/system/health_monitor.py');
    
    this.auraProcess = spawn('python3', [auraScriptPath], {
      cwd: path.join(__dirname, '../..'),
      stdio: ['pipe', 'pipe', 'pipe']
    });

    this.auraProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log('AURA Output:', output);
      
      // Parse health data from AURA output
      this.parseAURAOutput(output);
    });

    this.auraProcess.stderr.on('data', (data) => {
      console.error('AURA Error:', data.toString());
    });

    this.auraProcess.on('close', (code) => {
      console.log(`AURA process exited with code ${code}`);
      this.isConnected = false;
      this.io.emit('system_status', {
        type: 'aura_status',
        status: 'disconnected',
        timestamp: Date.now()
      });
    });

    this.auraProcess.on('error', (error) => {
      console.error('AURA process error:', error);
      this.isConnected = false;
    });

    this.isConnected = true;
    this.io.emit('system_status', {
      type: 'aura_status',
      status: 'connected',
      timestamp: Date.now()
    });
  }

  parseAURAOutput(output) {
    try {
      // Look for JSON data in the output
      const lines = output.split('\n');
      
      for (const line of lines) {
        if (line.includes('{') && line.includes('}')) {
          try {
            const data = JSON.parse(line);
            this.broadcastHealthData(data);
          } catch (e) {
            // Not JSON, continue
          }
        }
      }

      // Parse specific patterns from AURA output
      if (output.includes('ALERT')) {
        this.parseAlerts(output);
      }
      
      if (output.includes('FIRING')) {
        this.parseFiringEvents(output);
      }
      
    } catch (error) {
      console.error('Error parsing AURA output:', error);
    }
  }

  parseAlerts(output) {
    const alertRegex = /🟡 ALERT \[(\w+)\] (.+?): (.+)/g;
    let match;
    
    while ((match = alertRegex.exec(output)) !== null) {
      const [, component, message, value] = match;
      
      const alert = {
        id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        timestamp: Date.now(),
        severity: 'warning',
        component: component.toLowerCase(),
        message: `${message}: ${value}`,
        details: `Detected in AURA output`
      };
      
      this.io.emit('alert', alert);
    }
  }

  parseFiringEvents(output) {
    // This would parse firing events from AURA output
    // For now, we'll simulate some data
    if (output.includes('Processed data')) {
      const firingEvent = {
        timestamp: Date.now(),
        neuron_id: `neuron_${Math.floor(Math.random() * 100)}`,
        region: ['thalamus', 'hippocampus', 'amygdala'][Math.floor(Math.random() * 3)],
        firing_strength: Math.random(),
        trigger_source: ['attention', 'hormone', 'learning', 'input'][Math.floor(Math.random() * 4)],
        trigger_details: 'Simulated firing event',
        context: 'AURA processing'
      };
      
      this.io.emit('firing_event', firingEvent);
    }
  }

  broadcastHealthData(data) {
    // Broadcast health snapshot to all connected clients
    this.io.emit('health_snapshot', data);
  }

  start() {
    this.server.listen(this.port, () => {
      console.log(`🚀 AURA WebSocket server running on port ${this.port}`);
      console.log(`📡 Dashboard available at: http://localhost:3000`);
      
      // Start AURA monitor after server is running
      setTimeout(() => {
        this.startAURAMonitor();
      }, 1000);
    });
  }

  stop() {
    if (this.auraProcess) {
      this.auraProcess.kill();
    }
    this.server.close();
  }
}

// Start the server
const server = new AURAWebSocketServer();
server.start();

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down AURA WebSocket server...');
  server.stop();
  process.exit(0);
});

module.exports = AURAWebSocketServer;
