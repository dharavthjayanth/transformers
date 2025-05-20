import React, { useState, useEffect } from "react";
import Sidebar from "../components/Sidebar";
import ChatWindow from "../components/ChatWindow";
import { getScopesFromToken, getUserNameFromToken } from "../../utilities/token"
import { queryLLM } from "../../utilities/api";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [scopes, setScopes] = useState([]);
  const [userName, setUserName] = useState("User");
  const [theme, setTheme] = useState("light");

  useEffect(() => {
    setScopes(getScopesFromToken());
    setUserName(getUserNameFromToken());
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const addMessage = (msg) => {
    setMessages((prev) => [...prev, msg]);
    setTimeout(() => {
      setMessages((prev) => [...prev, { role: "bot", text: `📢 Sample response to: "${msg.text}"` }]);
    }, 500);
  };

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <Sidebar
        messages={messages}
        scopes={scopes}
        userName={userName}
        theme={theme}
        setTheme={setTheme}
      />
      <ChatWindow messages={messages} onSend={addMessage} />
    </div>
  );
};

export default Chatbot;
