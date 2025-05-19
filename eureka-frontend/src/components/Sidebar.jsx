import React from "react";
import { useNavigate } from "react-router-dom";
import "../styling/Sidebar.css"

const Sidebar = ({ messages, scopes, userName, theme, setTheme }) => {
  const navigate = useNavigate();
  const userQueries = messages.filter((m) => m.role === "user");

  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/login");
  };

  const toggleTheme = () => {
    setTheme(theme === "light" ? "dark" : "light");
  };

  return (
    <div className="sidebar">
      <div className="user-section">
        <div className="user-avatar">👤</div>
        <div className="user-name">{userName || "User"}</div>
      </div>

      <div>
        <h3>🔐 Access Levels</h3>
        <ul>
          {scopes.map((scope, idx) => (
            <li key={idx}>🔸 {scope}</li>
          ))}
        </ul>
      </div>

      <div>
        <h3>🕑 Previous Queries</h3>
        <ul>
          {userQueries.map((q, idx) => (
            <li key={idx}>🧑‍💬 {q.text}</li>
          ))}
        </ul>
      </div>

      <div className="theme-toggle-container">
        <button onClick={toggleTheme} className="theme-toggle-button">
          {theme === "light" ? "🌙 Dark Mode" : "☀️ Light Mode"}
        </button>
      </div>

      <div className="logout-container">
        <button onClick={handleLogout} className="logout-button">
          🚪 Logout
        </button>
      </div>
    </div>
  );
};

export default Sidebar;
