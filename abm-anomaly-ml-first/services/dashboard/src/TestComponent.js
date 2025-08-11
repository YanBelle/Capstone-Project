import React from 'react';

const TestComponent = () => {
  console.log('🔥 TestComponent is rendering!');
  
  return (
    <div style={{
      backgroundColor: 'red',
      color: 'white',
      padding: '20px',
      margin: '20px',
      fontSize: '18px',
      textAlign: 'center'
    }}>
      <h1>🚨 REACT IS WORKING! 🚨</h1>
      <p>If you can see this red box, React is rendering correctly.</p>
      <p>Current time: {new Date().toLocaleTimeString()}</p>
    </div>
  );
};

export default TestComponent;
