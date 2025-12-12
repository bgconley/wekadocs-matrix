GLiNER Service Management

  Start the Service

  cd /Users/brennanconley/vibecode/wekadocs-matrix/services/gliner-ner
  ./run.sh

  Options:
  ./run.sh                    # Auto-detect device (MPS preferred)
  ./run.sh --device cpu       # Force CPU mode
  ./run.sh --port 9003        # Use different port

  Background mode:
  nohup python3 server.py > /tmp/gliner-ner.log 2>&1 &

  Stop the Service

  pkill -f "server.py"

  Or if you know the PID:
  kill <PID>

  Check Status

  # Health check
  curl -s http://127.0.0.1:9002/healthz | python3 -m json.tool

  # Check if running
  pgrep -f "server.py" && echo "Running" || echo "Not running"

  View Logs (if started in background)

  tail -f /tmp/gliner-ner.log

  ---
  Current status: The service is running from our earlier testing. It will stop automatically when you close that terminal, or you can run pkill -f "server.py" to stop it now.
