import { createRoot } from 'react-dom/client';
import './styles/app.css';
import { PitwallPage } from './pages/pitwall';

createRoot(document.getElementById('root')!).render(<PitwallPage />
);
