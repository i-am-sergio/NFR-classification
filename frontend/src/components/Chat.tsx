import React, { useState, useRef, useEffect } from "react";
import Message from "./Message";
import ChatInput from "./ChatInput";
import Navbar from "./Navbar";
import { useAppContext } from "../context/AppContext";

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<
    { text: string; sender: "user" | "bot" }[]
  >([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { setMessage, sendData } = useAppContext();

  const handleSendMessage = async (message: string) => {
    setMessages([...messages, { text: message, sender: "user" }]);
    setMessage(message); // Actualiza el mensaje en el contexto
    setMessages((prev) => [...prev, { text: "Pensando...", sender: "bot" }]);

    try {
      const response = await sendData();
      if (response) {
        setMessages((prev) => prev.filter((msg) => msg.text !== "Pensando..."));
        setMessages((prev) => [...prev, { text: response, sender: "bot" }]); // Add Response
      } else {
        setMessages((prev) => prev.filter((msg) => msg.text !== "Pensando..."));
        setMessages((prev) => [
          ...prev,
          { text: "Error al obtener respuesta", sender: "bot" },
        ]);
      }
    } catch (error) {
      console.error("Error al obtener la respuesta:", error);
      setMessages((prev) => prev.filter((msg) => msg.text !== "Pensando..."));
      setMessages((prev) => [
        ...prev,
        { text: "Error al obtener respuesta", sender: "bot" },
      ]);
    }
  };
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="flex flex-col w-full h-screen">
      <div className="w-full">
        <div className="max-w-4xl mx-auto">
          <Navbar />
        </div>
      </div>
      <div className="flex-1 w-full overflow-hidden">
        <div className="w-full px-4 h-full overflow-y-auto scrollbar-thumb-gray-300 scrollbar-track-transparent scrollbar-thin">
          <div className="max-w-4xl mx-auto space-y-4 flex flex-col p-4 whitespace-normal h-full">
            {messages.length === 0 ? ( // Condición para el mensaje de bienvenida
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-gray-500 text-3xl px-40">
                  Hola, Te ayudare con la clasificación de tus{" "}
                  <span className="text-violet-500 font-semibold text-3xl">
                    Requisitos{" "}
                  </span>
                  !!!
                </div>
              </div>
            ) : (
              <>
                {messages.map((msg, index) => (
                  <Message key={index} text={msg.text} sender={msg.sender} />
                ))}
                <div ref={messagesEndRef} />
              </>
            )}
          </div>
        </div>
      </div>
      <div className="w-full">
        <div className="max-w-4xl mx-auto">
          <ChatInput onSend={handleSendMessage} />
        </div>
      </div>
    </div>
  );
};

export default Chat;
