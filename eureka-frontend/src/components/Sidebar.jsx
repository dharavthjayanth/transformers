// src/components/Sidebar.jsx
import React from "react";

const Sidebar = ({ messages, scopes }) => {
    return (
        <div style={{ width: "250px", background: "#f4f4f4", padding: "1rem" }}>
            <h3>Access Levels</h3>
            <ul>
                {scopes.map((scope, idx) => (
                    <li key={idx}>🔐 {scope}</li>
                ))}
            </ul>

            <h3 style={{ marginTop: "2rem" }}>Previous Queries</h3>
            <ul>
                {messages.map((m, idx) =>
                    m.role === "user" ? <li key={idx}>🧑 {m.text}</li> : null
                )}
            </ul>
        </div>
    );
};

export default Sidebar;
