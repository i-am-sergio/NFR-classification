import React, { useState } from "react";
import { IoMdArrowDropdown } from "react-icons/io";
import { useAppContext } from "../context/AppContext";

const models = [
  "LR Model",
  "SVM Model",
  "KNN Model",
  "MNB Model",
  "RF Model",
  "CATBOOST Model",
];

const languages = ["English", "Spanish", "Portuguese"];

const Navbar: React.FC = () => {

  const { selectedModel, setSelectedModel, selectedLanguage, setSelectedLanguage } = useAppContext();
  
  // const [selectedModel, setSelectedModel] = useState<string>("LR Model");
  // const [selectedLanguage, setSelectedLanguage] = useState<string>("English");
  
  const [isOpenModel, setIsOpenModel] = useState(false); 
  const [isOpenLanguage, setIsOpenLanguage] = useState(false);

  const handleModelChange = (model: string) => {
    setSelectedModel(model);
    setIsOpenModel(false);
  };

  const handleLanguageChange = (language: string) => {
    setSelectedLanguage(language);
    setIsOpenLanguage(false);
  };

  return (
    <nav className="bg-transparent p-4 text-white">
      <div className="container mx-auto flex justify-between items-center">
        <span className="font-bold text-2xl sigmar-regular">ClassifyRQ</span>
        <div className="flex items-center">
          {/* Model Selector */}
          <div className="relative mr-4">
            <button
              className="text-white font-bold rounded-md hover:cursor-pointer px-3 py-2 transition-colors duration-500 flex items-center"
              onClick={() => setIsOpenModel(!isOpenModel)}
            >
              {selectedModel}
              <IoMdArrowDropdown className="ml-2" />
            </button>
            {isOpenModel && (
              <div className="absolute right-0 mt-2 w-44 rounded-md shadow-lg bg-[#27292b] z-10">
                {models.map((model) => (
                  <div
                    key={model}
                    className="px-4 py-2 text-left text-white cursor-pointer hover:bg-[#4d4f52]"
                    onClick={() => handleModelChange(model)}
                  >
                    {model}
                  </div>
                ))}
              </div>
            )}
          </div>
            {/* Language Selector */}
          <div className="relative">
            <button
              className="text-white font-bold rounded-md hover:cursor-pointer px-3 py-2 transition-colors duration-500 flex items-center"
              onClick={() => setIsOpenLanguage(!isOpenLanguage)}
            >
              {selectedLanguage}
              <IoMdArrowDropdown className="ml-2" />
            </button>
            {isOpenLanguage && (
              <div className="absolute right-0 mt-2 w-32 rounded-md shadow-lg bg-[#27292b] z-10">
                {languages.map((language) => (
                  <div
                    key={language}
                    className="px-4 py-2 text-left text-white cursor-pointer hover:bg-[#4d4f52]"
                    onClick={() => handleLanguageChange(language)}
                  >
                    {language}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;