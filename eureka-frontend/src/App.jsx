// src/App.jsx
import React, { useState } from 'react';
import Login from './login';
import Chat from './chat';

function App() {
  const [user, setUser] = useState(null);

  return (
    <div className="app">
      {user ? <Chat user={user} /> : <Login setUser={setUser} />}
    </div>
  );
}

export default App;
