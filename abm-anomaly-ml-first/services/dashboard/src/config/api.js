// Simple API configuration for local development
class ApiConfig {
  constructor() {
    // Use localhost for development (will be proxied through nginx)
    this.baseUrl = 'http://localhost';
  }

  getApiUrl() {
    return `${this.baseUrl}/api`;
  }

  // Utility method to construct API endpoints
  endpoint(path = '') {
    // Simply append the path to baseUrl, ensuring proper format
    if (!path) {
      return `${this.baseUrl}/api`;
    }
    
    // Remove leading slashes from path
    const cleanPath = path.replace(/^\/+/, '');
    
    // Return the full URL
    return `${this.baseUrl}/${cleanPath}`;
  }
}

// Create a singleton instance
const apiConfig = new ApiConfig();

export default apiConfig;
