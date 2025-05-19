import { jwtDecode } from "jwt-decode";

export const getUserNameFromToken = () => {
  const token = localStorage.getItem("token");
  if (!token) return "User";

  try {
    const decoded = jwtDecode(token);
    return decoded.name || "User";
  } catch {
    return "User";
  }
};

export const getScopesFromToken = () => {
  const token = localStorage.getItem("token");
  if (!token) return [];

  try {
    const decoded = jwtDecode(token);
    return decoded.scopes || [];
  } catch {
    return [];
  }
};