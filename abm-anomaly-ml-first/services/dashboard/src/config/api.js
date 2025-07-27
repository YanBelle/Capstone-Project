// Simple API configuration for local development
class ApiConfig {
  constructor() {
    // Force localhost for development  
    this.baseUrl = 'http://localhost';
  }

  getApiUrl() {
    return `${this.baseUrl}/api`;
  }

  // Utility method to construct API endpoints
  endpoint(path = '') {
    // Remove leading slash and /api prefix if present to avoid duplication
    const cleanPath = path.replace(/^\/+/, '').replace(/^api\/+/, '');
    
    if (!cleanPath) {
      return `${this.baseUrl}/api`;
    }
    
    // Always ensure we use the correct format: http://localhost/api/v1/...
    return `${this.baseUrl}/api/${cleanPath}`;
  }
}

// Create a singleton instance
const apiConfig = new ApiConfig();

export default apiConfig;
