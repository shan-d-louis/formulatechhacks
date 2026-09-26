import { createRoot } from 'react-dom/client';
import './styles/app.css';
import { HomePage } from './pages/index';

createRoot(document.getElementById('root')!).render(<HomePage />
);
