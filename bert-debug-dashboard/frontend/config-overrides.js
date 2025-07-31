module.exports = function override(config, env) {
  // Disable WebSocket in development
  if (env === 'development') {
    config.devServer = config.devServer || {};
    config.devServer.hot = false;
    config.devServer.liveReload = false;
    config.devServer.webSocketServer = false;
    config.devServer.client = {
      webSocketTransport: 'sockjs',
      webSocketURL: 'auto://0.0.0.0:0/ws'
    };
  }
  
  return config;
};
