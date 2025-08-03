const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  // Disable WebSocket proxy
  app.use(
    '/ws',
    createProxyMiddleware({
      target: 'http://localhost:3000',
      ws: false, // Disable WebSocket proxying
      changeOrigin: true,
      onError: (err, req, res) => {
        res.status(500).send('WebSocket disabled');
      }
    })
  );
  
  // Proxy API requests to backend
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://backend:8000',
      changeOrigin: true,
      ws: false,
    })
  );
};
