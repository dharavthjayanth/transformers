import { jwtDecode } from "jwt-decode";

export const getScopesFromToken = () => {
    const token = localStorage.getItem("token");
    if (!token) return [];

    try {
        const decoded = jwtDecode(token);
        return decoded.scopes || [];
    } catch (err) {
        console.error("Invalid token:", err);
        return [];
    }
};