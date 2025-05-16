import React, { useState } from "react";

const ChatWindow = ({ messages, onSend }) => {
    const [input, setInput] = useState("");

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!input.trim()) return;
        onSend({ role: "user", text: input });
        setInput("");
    };

    return (
        <div style={{ flex: 1, padding: "1rem", display: "flex", flexDirection: "column" }}>
            <div style={{ flex: 1, overflowY: "auto", marginBottom: "1rem" }}>
                {messages.map((msg, idx) => (
                    <div key={idx} style={{ margin: "0.5rem 0" }}>
                        <strong style={{ color: msg.role === "user" ? "#333" : "#007acc" }}>
                            {msg.role === "user" ? "🧑 You" : "🤖 Bot"}:
                        </strong>{" "}
                        {msg.text}
                    </div>
                ))}
            </div>
            <form onSubmit={handleSubmit} style={{ display: "flex" }}>
                <input
                    type="text"
                    value={input}
                    placeholder="Type your message..."
                    onChange={(e) => setInput(e.target.value)}
                    style={{ flex: 1, padding: "0.5rem" }}
                />
                <button type="submit" style={{ padding: "0.5rem 1rem" }}>Send</button>
            </form>
        </div>
    );
};

export default ChatWindow;
