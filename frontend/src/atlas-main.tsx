import { createRoot } from 'react-dom/client';
import './styles/app.css';
import { AtlasPage } from './pages/atlas';

createRoot(document.getElementById('root')!).render(<AtlasPage />
);
