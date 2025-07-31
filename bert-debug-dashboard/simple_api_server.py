#!/usr/bin/env python3

# Simple test to verify the backend API works
import http.server
import socketserver
import json
from urllib.parse import urlparse, parse_qs
import time

class APIHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"message": "Simple API working", "timestamp": time.time()}
            self.wfile.write(json.dumps(response).encode())
        
        elif self.path == '/api/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"status": "healthy", "model_loaded": True}
            self.wfile.write(json.dumps(response).encode())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/api/analyze':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Mock response
            response = {
                "text": "mock analysis",
                "tokens": ["mock", "tokens"],
                "predicted_class": 1,
                "probabilities": [0.1, 0.7, 0.15, 0.05],
                "attention_weights": [],
                "token_importance": [0.5, 0.5],
                "hidden_states": {"cls_embeddings": []},
                "analysis_time": "0.1s"
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

if __name__ == "__main__":
    PORT = 8000
    with socketserver.TCPServer(("", PORT), APIHandler) as httpd:
        print(f"Simple API server running on port {PORT}")
        print(f"Test at: http://localhost:{PORT}")
        print(f"Health check: http://localhost:{PORT}/api/health")
        httpd.serve_forever()
