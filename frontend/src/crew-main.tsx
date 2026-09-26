import { createRoot } from 'react-dom/client';
import './styles/app.css';
import { CrewPage } from './pages/crew';

document.body.className = 'l0';
createRoot(document.getElementById('root')!).render(<CrewPage />
);
