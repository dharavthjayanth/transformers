import React, { useState, useEffect } from "react";
import Sidebar from "../components/Sidebar";
import ChatWindow from "../components/ChatWindow";
import { getScopesFromToken } from "../../utilities/token";

const Chatbot = () => {
    const [messages, setMessages] = useState([]);
    const [scopes, setScopes] = useState([]);

    useEffect(() => {
        const userScopes = getScopesFromToken();
        setScopes(userScopes);
    }, []);

    const addMessage = (msg) => {
        setMessages((prev) => [...prev, msg]);
    };

    return (
        <div style={{ display: "flex", height: "100vh" }}>
            <Sidebar messages={messages} scopes={scopes} />
            <ChatWindow messages={messages} onSend={addMessage} />
        </div>
    );
};

export default Chatbot;
