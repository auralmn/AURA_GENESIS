#!/bin/bash

echo "🚀 Starting AURA Health Dashboard..."
echo "=================================="

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Please run this script from the dashboard directory"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Next.js dependencies..."
    npm install
fi

# Install WebSocket server dependencies
if [ ! -d "server/node_modules" ]; then
    echo "📦 Installing WebSocket server dependencies..."
    cd server
    npm install
    cd ..
fi

# Start WebSocket server in background
echo "🔌 Starting WebSocket server..."
cd server
node websocket-server.js &
WS_PID=$!
cd ..

# Wait a moment for server to start
sleep 2

# Start Next.js development server
echo "🌐 Starting Next.js dashboard..."
npm run dev &
NEXT_PID=$!

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down dashboard..."
    kill $WS_PID 2>/dev/null
    kill $NEXT_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo ""
echo "✅ Dashboard started successfully!"
echo "📡 WebSocket server: http://localhost:3001"
echo "🌐 Dashboard: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for processes
wait
