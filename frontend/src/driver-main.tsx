import { createRoot } from 'react-dom/client';
import './styles/app.css';
import { DriverPage } from './pages/driver';

document.body.className = '';
createRoot(document.getElementById('root')!).render(<DriverPage />
);
