import { Routes, Route, Navigate } from "react-router-dom";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Chatbot from "./pages/Chatbot";
import PrivateRoute from '../authentication/PrivateRoute';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/login" />} />
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route
        path="/chatbot"
        element={
          <PrivateRoute>
            <Chatbot />
          </PrivateRoute>
        }
      />
    </Routes>
  );
}

export default App;
