// Universe Portal - Interactive Features

class UniversePortal {
    constructor() {
        this.ws = null;
        this.eventStream = document.getElementById('event-stream');
        this.reconnectInterval = 5000;
        this.maxEvents = 100;
        
        this.initWebSocket();
        this.setupEventHandlers();
        this.startAnimations();
    }
    
    initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('Connected to Universe stream');
            this.addEvent('Connected to universe stream', 'system');
        };
        
        this.ws.onmessage = (event) => {
            try {
                const update = JSON.parse(event.data);
                this.handleUniverseUpdate(update);
            } catch (e) {
                console.error('Failed to parse update:', e);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.addEvent('Connection error', 'error');
        };
        
        this.ws.onclose = () => {
            console.log('Disconnected from Universe stream');
            this.addEvent('Disconnected - reconnecting...', 'system');
            setTimeout(() => this.initWebSocket(), this.reconnectInterval);
        };
    }
    
    handleUniverseUpdate(update) {
        switch (update.type) {
            case 'TickAdvanced':
                this.updateTick(update.tick, update.year);
                break;
                
            case 'CivilizationProgress':
                this.handleCivilizationProgress(update);
                break;
                
            case 'ExistentialDiscovery':
                this.handleExistentialDiscovery(update);
                break;
                
            case 'CosmicEvent':
                this.handleCosmicEvent(update);
                break;
                
            case 'AgentPetition':
                this.handleAgentPetition(update);
                break;
                
            default:
                console.log('Unknown update type:', update.type);
        }
    }
    
    updateTick(tick, year) {
        // Update tick counter with smooth animation
        const tickElement = document.querySelector('.status-card:nth-child(1) .value');
        const yearElement = document.querySelector('.status-card:nth-child(1) .label');
        
        if (tickElement) {
            this.animateNumber(tickElement, parseInt(tickElement.textContent), tick);
        }
        
        if (yearElement) {
            yearElement.textContent = `Year: ${this.formatScientific(year)}`;
        }
    }
    
    handleCivilizationProgress(update) {
        const message = `Civilization ${update.id} reached ${update.progress.toFixed(1)}% of ${update.milestone}`;
        this.addEvent(message, 'progress');
        
        // Update milestone visualization if on the correct page
        if (update.milestone === 'Philosophical Enlightenment' && update.progress > 50) {
            this.highlightMilestone('philosophy');
        }
    }
    
    handleExistentialDiscovery(update) {
        const { discovery, civilization_id } = update;
        const message = `🌟 ${civilization_id} discovered: "${discovery.title}" - ${discovery.description}`;
        this.addEvent(message, 'discovery');
        
        // Special effects for major discoveries
        if (discovery.category === 'ExistentialTruth') {
            this.triggerEnlightenmentEffect();
        }
        
        // Show implications
        if (discovery.implications && discovery.implications.length > 0) {
            discovery.implications.forEach(impl => {
                setTimeout(() => {
                    this.addEvent(`↳ ${impl}`, 'discovery');
                }, 500);
            });
        }
    }
    
    handleCosmicEvent(update) {
        const message = `🌌 ${update.description} at (${update.location.x.toFixed(1)}, ${update.location.y.toFixed(1)}, ${update.location.z.toFixed(1)})`;
        this.addEvent(message, 'cosmic');
    }
    
    handleAgentPetition(update) {
        const message = `📨 Petition from ${update.from}: "${update.content}"`;
        this.addEvent(message, 'petition');
        
        // Check if it's an existential question
        if (update.content.toLowerCase().includes('why') || 
            update.content.toLowerCase().includes('purpose') ||
            update.content.toLowerCase().includes('meaning')) {
            this.addEvent('↳ Existential query detected', 'philosophical');
        }
    }
    
    addEvent(message, type = 'default') {
        const eventDiv = document.createElement('div');
        eventDiv.className = `event ${type}`;
        eventDiv.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        
        this.eventStream.appendChild(eventDiv);
        
        // Keep only the latest events
        while (this.eventStream.children.length > this.maxEvents) {
            this.eventStream.removeChild(this.eventStream.firstChild);
        }
        
        // Scroll to bottom
        this.eventStream.scrollTop = this.eventStream.scrollHeight;
    }
    
    animateNumber(element, start, end) {
        const duration = 1000;
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const current = Math.floor(start + (end - start) * this.easeOutCubic(progress));
            element.textContent = current.toLocaleString();
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }
    
    easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }
    
    formatScientific(num) {
        if (num < 1e6) return num.toFixed(0);
        return num.toExponential(2);
    }
    
    highlightMilestone(type) {
        const milestones = document.querySelectorAll('.milestone');
        milestones.forEach(milestone => {
            if (milestone.querySelector('.label').textContent.toLowerCase().includes(type)) {
                milestone.classList.add('highlight');
                setTimeout(() => milestone.classList.remove('highlight'), 3000);
            }
        });
    }
    
    triggerEnlightenmentEffect() {
        // Create a cosmic enlightenment visual effect
        const effect = document.createElement('div');
        effect.className = 'enlightenment-burst';
        effect.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 0;
            height: 0;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(155,89,182,0.8) 0%, transparent 70%);
            z-index: 9999;
            pointer-events: none;
        `;
        
        document.body.appendChild(effect);
        
        // Animate the burst
        effect.animate([
            { width: '0px', height: '0px', opacity: 1 },
            { width: '2000px', height: '2000px', opacity: 0 }
        ], {
            duration: 3000,
            easing: 'ease-out'
        }).onfinish = () => effect.remove();
        
        // Play a sound if available
        this.playEnlightenmentSound();
    }
    
    playEnlightenmentSound() {
        // Create a simple tone using Web Audio API
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.setValueAtTime(440, audioContext.currentTime);
        oscillator.frequency.exponentialRampToValueAtTime(880, audioContext.currentTime + 0.5);
        
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 1);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 1);
    }
    
    setupEventHandlers() {
        // Handle navigation clicks
        document.querySelectorAll('nav a').forEach(link => {
            link.addEventListener('click', (e) => {
                if (link.getAttribute('href').startsWith('/')) {
                    // Internal navigation - could implement SPA routing here
                    console.log('Navigate to:', link.getAttribute('href'));
                }
            });
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K for quick search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.openQuickSearch();
            }
            
            // ESC to close modals
            if (e.key === 'Escape') {
                this.closeAllModals();
            }
        });
    }
    
    openQuickSearch() {
        console.log('Quick search not implemented yet');
        // TODO: Implement quick search modal
    }
    
    closeAllModals() {
        // Close any open modals
        document.querySelectorAll('.modal').forEach(modal => {
            modal.style.display = 'none';
        });
    }
    
    startAnimations() {
        // Animate milestone icons on hover
        document.querySelectorAll('.milestone').forEach(milestone => {
            milestone.addEventListener('mouseenter', () => {
                milestone.querySelector('.icon').style.transform = 'scale(1.2)';
            });
            
            milestone.addEventListener('mouseleave', () => {
                milestone.querySelector('.icon').style.transform = 'scale(1)';
            });
        });
        
        // Periodic pulse animation for active elements
        setInterval(() => {
            document.querySelectorAll('.milestone.partial .icon').forEach(icon => {
                icon.animate([
                    { transform: 'scale(1)', boxShadow: '0 0 20px var(--warning)' },
                    { transform: 'scale(1.1)', boxShadow: '0 0 40px var(--warning)' },
                    { transform: 'scale(1)', boxShadow: '0 0 20px var(--warning)' }
                ], {
                    duration: 1000,
                    easing: 'ease-in-out'
                });
            });
        }, 3000);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.universePortal = new UniversePortal();
});

// Export for use in other scripts
window.handleUniverseUpdate = (update) => {
    if (window.universePortal) {
        window.universePortal.handleUniverseUpdate(update);
    }
};