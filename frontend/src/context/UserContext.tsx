"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { User } from "@/types/user";

interface UserContextType {
  activeUser: User | null;
  setActiveUser: (user: User | null) => void;
  isLoading: boolean;
}

const UserContext = createContext<UserContextType>({
  activeUser: null,
  setActiveUser: () => {},
  isLoading: true,
});

export const UserProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [activeUser, setActiveUserState] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    try {
      const stored = localStorage.getItem("active_user");
      if (stored) {
        setActiveUserState(JSON.parse(stored));
      }
    } catch {
      // Ignore JSON parse errors
    } finally {
      setIsLoading(false);
    }

  }, []);

  const setActiveUser = (user: User | null) => {
    setActiveUserState(user);
    if (user) {
      localStorage.setItem("active_user", JSON.stringify(user));
    } else {
      localStorage.removeItem("active_user");
    }
  };

  return (
    <UserContext.Provider value={{ activeUser, setActiveUser, isLoading }}>
      {children}
    </UserContext.Provider>
  );
};

export const useUser = () => useContext(UserContext);
