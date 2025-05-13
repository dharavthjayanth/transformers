import React, { useState } from 'react';
import axios from 'axios';
import './styles.css';

function Login({ setUser }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleLogin = async () => {
    try {
      const res = await axios.post('http://localhost:5000/query', {
        username,
        password,
        query: 'Test login access'  // dummy query just to authenticate
      });

      if (res.status === 200) {
        setUser({ username, password });
      }
    } catch (err) {
      setError('❌ Invalid credentials or server error');
    }
  };

  return (
    <div className="login-container">
      <h2>Eureka AI Assistant</h2>
      <input
        type="text"
        placeholder="Username"
        value={username}
        onChange={e => setUsername(e.target.value)}
      />
      <input
        type="password"
        placeholder="Password"
        value={password}
        onChange={e => setPassword(e.target.value)}
      />
      <button onClick={handleLogin}>Login</button>
      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default Login;
