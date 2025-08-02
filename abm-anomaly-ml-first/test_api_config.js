class ApiConfig {
  constructor() {
    this.baseUrl = 'http://localhost';
  }

  endpoint(path = '') {
    if (!path) {
      return `${this.baseUrl}/api`;
    }
    
    const cleanPath = path.replace(/^\/+/, '');
    return `${this.baseUrl}/${cleanPath}`;
  }
}

const apiConfig = new ApiConfig();
console.log('Testing API config:');
console.log('Input: api/v1/bert/analyze');
console.log('Output:', apiConfig.endpoint('api/v1/bert/analyze'));
console.log('');
console.log('Input: /api/v1/bert/analyze');
console.log('Output:', apiConfig.endpoint('/api/v1/bert/analyze'));
console.log('');
console.log('Expected: http://localhost/api/v1/bert/analyze');
