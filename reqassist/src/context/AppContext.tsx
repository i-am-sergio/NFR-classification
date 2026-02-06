import React, { createContext, useState, useContext } from 'react';
import axios from 'axios';

interface AppContextProps {
  message: string;
  setMessage: React.Dispatch<React.SetStateAction<string>>;
  selectedModel: string;
  setSelectedModel: React.Dispatch<React.SetStateAction<string>>;
  selectedLanguage: string;
  setSelectedLanguage: React.Dispatch<React.SetStateAction<string>>;
  sendData: () => Promise<string | null>;
}

const AppContext = createContext<AppContextProps | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [message, setMessage] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('LR Model');
  const [selectedLanguage, setSelectedLanguage] = useState<string>('English');

  const sendData = async (): Promise<string | null> => {
    try {
      const response = await axios.post('http://localhost:5000/predict12', {
        message,
        selectedModel,
        selectedLanguage,
      });
      console.log('Response from server:', response.data);
      const classPrediction = response.data.predictionResult
      return classPrediction;
    } catch (error) {
      console.error('Error sending data:', error);
      return null;
    }
  };

  const value: AppContextProps = {
    message,
    setMessage,
    selectedModel,
    setSelectedModel,
    selectedLanguage,
    setSelectedLanguage,
    sendData,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
};