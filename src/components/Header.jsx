import React from 'react';
import { Bell, Search, Moon, Sun, User } from 'lucide-react';
import './Header.css';

const Header = ({ title }) => {
    return (
        <header className="header">
            <div className="header-left">
                <h1 className="header-title">{title}</h1>
            </div>

            <div className="header-search">
                <Search className="search-icon" size={16} strokeWidth={1.5} />
                <input type="text" placeholder="Search..." />
            </div>

            <div className="header-actions">
                <button className="action-btn">
                    <Bell size={18} strokeWidth={1.5} />
                    <span className="notification-dot"></span>
                </button>
                <button className="action-btn">
                    <Moon size={18} strokeWidth={1.5} />
                </button>
                <div className="header-divider"></div>
                <button className="action-btn user-profile">
                    <User size={18} strokeWidth={1.5} />
                </button>
            </div>
        </header>
    );
};

export default Header;
