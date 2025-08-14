// Test script to verify our API configuration logic
const testApiConfig = () => {
  console.log('Testing API Configuration Logic:');
  
  // Simulate production environment
  const mockProdEnv = { NODE_ENV: 'production' };
  const prodBaseUrl = mockProdEnv.NODE_ENV === 'production' ? '' : 'http://localhost:8000';
  console.log('Production environment - baseUrl:', prodBaseUrl);
  console.log('Production endpoint example:', `${prodBaseUrl}/api/v1/sessions`);
  
  // Simulate development environment  
  const mockDevEnv = { NODE_ENV: 'development' };
  const devBaseUrl = mockDevEnv.NODE_ENV === 'production' ? '' : 'http://localhost:8000';
  console.log('Development environment - baseUrl:', devBaseUrl);
  console.log('Development endpoint example:', `${devBaseUrl}/api/v1/sessions`);
  
  // Test the endpoint method logic
  const testEndpoint = (baseUrl, path) => {
    const cleanPath = path.replace(/^\/+/, '');
    return `${baseUrl}/${cleanPath}`;
  };
  
  console.log('\nTesting endpoint method:');
  console.log('Production:', testEndpoint('', 'api/v1/sessions'));
  console.log('Development:', testEndpoint('http://localhost:8000', 'api/v1/sessions'));
  console.log('Production cash:', testEndpoint('', 'api/cash-forecasting/terminal-status'));
  console.log('Development cash:', testEndpoint('http://localhost:8000', 'api/cash-forecasting/terminal-status'));
};

testApiConfig();
