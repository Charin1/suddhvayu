import React, { createContext, useContext, useState, ReactNode } from 'react';

// India → State → City hierarchy
// For now, we focus on Gujarat, but structure supports expansion
const INDIA_STATES = {
    "Gujarat": ["Ahmedabad", "Gandhinagar", "Surat", "Rajkot", "Vadodara"]
};

interface LocationContextType {
    selectedState: string;
    selectedCity: string;
    startDate: string;
    endDate: string;
    availableCities: string[];
    setSelectedState: (state: string) => void;
    setSelectedCity: (city: string) => void;
    setStartDate: (date: string) => void;
    setEndDate: (date: string) => void;
}

const LocationContext = createContext<LocationContextType | undefined>(undefined);

interface LocationProviderProps {
    children: ReactNode;
}

export const LocationProvider: React.FC<LocationProviderProps> = ({ children }) => {
    const [selectedState, setSelectedState] = useState("Gujarat");
    const [selectedCity, setSelectedCity] = useState("Ahmedabad");
    const [startDate, setStartDate] = useState("2015-01-01");
    const [endDate, setEndDate] = useState("2020-12-31");

    const availableCities = INDIA_STATES[selectedState as keyof typeof INDIA_STATES] || [];

    const handleStateChange = (state: string) => {
        setSelectedState(state);
        // Auto-select first city when state changes
        const cities = INDIA_STATES[state as keyof typeof INDIA_STATES] || [];
        if (cities.length > 0) {
            setSelectedCity(cities[0]);
        }
    };

    return (
        <LocationContext.Provider
            value={{
                selectedState,
                selectedCity,
                startDate,
                endDate,
                availableCities,
                setSelectedState: handleStateChange,
                setSelectedCity,
                setStartDate,
                setEndDate,
            }}
        >
            {children}
        </LocationContext.Provider>
    );
};

export const useLocation = (): LocationContextType => {
    const context = useContext(LocationContext);
    if (!context) {
        throw new Error('useLocation must be used within a LocationProvider');
    }
    return context;
};

export const AVAILABLE_STATES = Object.keys(INDIA_STATES);
