import React from "react";
import { IoMdSend } from "react-icons/io";
import { GoPaperclip } from "react-icons/go";
import { useAppContext } from "../context/AppContext";

type ChatInputProps = {
  onSend: (message: string) => void;
};

const ChatInput: React.FC<ChatInputProps> = ({ onSend }) => {
  const { message, setMessage, sendData } = useAppContext();

  const handleSend = () => {
    if (message.trim()) {
      onSend(message);
      sendData();
      setMessage("");
    }
  };

  const handleUploadFile = () => {
    console.log("Subir Lista de Requisitos");
  };

  const isSendButtonDisabled = message.trim() === ""; // Estado para desactivar el botón

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter" && !isSendButtonDisabled) {
      event.preventDefault();
      handleSend();
    }
  };
  
  return (
    <div className="flex items-center p-4 bg-transparent border rounded-full border-gray-400 my-8">
      <div className="flex-1 rounded-full bg-transparent shadow-sm overflow-hidden flex items-center">
        <button
          className="p-3 text-white bg-transparent rounded-full shadow-md hover:bg-gray-700 hover:cursor-pointer flex items-center justify-center w-12 h-12"
          onClick={handleUploadFile}
        >
          <GoPaperclip className="w-8 h-8" />
        </button>
        <input
          type="text"
          className="text-white text-xl flex-1 p-3 focus:outline-none focus:ring-0 border-none bg-transparent pl-4 pr-2"
          placeholder="Escribe un mensaje..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          className={`p-3 text-white rounded-full shadow-md flex items-center justify-center w-12 h-12 ${
            isSendButtonDisabled
              ? "bg-gray-500 cursor-not-allowed"
              : "bg-violet-500 hover:cursor-pointer"
          }`}
          onClick={handleSend}
          disabled={isSendButtonDisabled} // Desactiva el botón
        >
          <IoMdSend className="w-8 h-8" />
        </button>
      </div>
    </div>
  );
};

export default ChatInput;