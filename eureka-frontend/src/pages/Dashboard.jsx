import { useNavigate } from 'react-router-dom';
import '../styling/Dashboard.css';

function Dashboard() {
    const navigate = useNavigate();

    const handleLogout = () => {
        // Clear auth logic here (e.g., localStorage/session)
        navigate('/login');
    };

    return (
        <div className="dashboard-container">
            <header className="dashboard-header">
                <h1>Welcome to your Dashboard</h1>
                <button onClick={handleLogout}>Logout</button>
            </header>

            <section className="dashboard-content">
                <div className="card">
                    <h3>Total Users</h3>
                    <p>128</p>
                </div>
                <div className="card">
                    <h3>Active Sessions</h3>
                    <p>17</p>
                </div>
                <div className="card">
                    <h3>Messages</h3>
                    <p>452</p>
                </div>
            </section>
        </div>
    );
}

export default Dashboard;
