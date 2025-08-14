// Test API configuration behavior in different environments
const testApiConfig = () => {
  console.log('=== API Configuration Test ===');
  
  // Test 1: Simulate production environment (NODE_ENV=production)
  console.log('\n1. Testing Production Environment:');
  const prodConfig = {
    baseUrl: process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000'
  };
  console.log('NODE_ENV:', process.env.NODE_ENV);
  console.log('baseUrl:', prodConfig.baseUrl);
  
  const endpoint = (baseUrl, path) => {
    const cleanPath = path.replace(/^\/+/, '');
    return `${baseUrl}/${cleanPath}`;
  };
  
  console.log('Session endpoint:', endpoint(prodConfig.baseUrl, 'api/v1/sessions'));
  console.log('Cash forecasting endpoint:', endpoint(prodConfig.baseUrl, 'api/cash-forecasting/terminal-status'));
  
  // Test 2: Simulate development environment
  console.log('\n2. Testing Development Environment:');
  process.env.NODE_ENV = 'development';
  const devConfig = {
    baseUrl: process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000'
  };
  console.log('NODE_ENV:', process.env.NODE_ENV);
  console.log('baseUrl:', devConfig.baseUrl);
  console.log('Session endpoint:', endpoint(devConfig.baseUrl, 'api/v1/sessions'));
  console.log('Cash forecasting endpoint:', endpoint(devConfig.baseUrl, 'api/cash-forecasting/terminal-status'));
  
  // Test 3: Current environment detection
  console.log('\n3. Current Environment Detection:');
  console.log('Current NODE_ENV:', process.env.NODE_ENV);
  console.log('Should use port 8000 for dev:', process.env.NODE_ENV !== 'production');
};

testApiConfig();
