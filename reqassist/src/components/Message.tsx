import React from "react";

type MessageProps = {
  text: string;
  sender: "user" | "bot";
};

const Message: React.FC<MessageProps> = ({ text, sender }) => {
  return (
    <div className={`flex ${sender === "user" ? "justify-end" : "justify-start"} w-full`}>
      <div className={`max-w-[80%] p-4 rounded-2xl shadow-md whitespace-pre-wrap break-words ${ // Añadido break-words
        sender === "user"
          ? "bg-[#343638] text-white rounded-br-none"
          : "bg-gray-200 text-black rounded-bl-none"
      }`}>
        {text}
      </div>
    </div>
  );
};

export default Message;