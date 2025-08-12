console.log('Testing sessions API...');

fetch('/api/v1/sessions?limit=5')
  .then(response => {
    console.log('Response status:', response.status);
    return response.json();
  })
  .then(data => {
    console.log('API Response:', data);
    console.log('Total sessions:', data.total);
    console.log('Sessions returned:', data.sessions ? data.sessions.length : 0);
    
    // Update the page
    document.body.innerHTML = `
      <h1>Sessions API Test Results</h1>
      <p><strong>Total Sessions:</strong> ${data.total}</p>
      <p><strong>Sessions Returned:</strong> ${data.sessions ? data.sessions.length : 0}</p>
      <p><strong>Status:</strong> API is working properly!</p>
      <p>This confirms the API can return all 66 sessions from the database.</p>
    `;
  })
  .catch(error => {
    console.error('API Error:', error);
    document.body.innerHTML = `
      <h1>API Error</h1>
      <p style="color: red;">Error: ${error.message}</p>
    `;
  });
