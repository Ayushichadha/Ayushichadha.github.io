// Floating Neural Network Structures - Multiple Networks Floating
class FloatingNeuralNetworks {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.networks = [];
    this.networkCount = 8;
    this.animationId = null;
    
    if (!this.ctx) {
      console.error('Could not get 2d context from canvas');
      return;
    }
    
    this.resize();
    this.init();
    this.animate();
    
    window.addEventListener('resize', () => this.resize());
  }
  
  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
    // Reinitialize networks after resize
    if (this.networks.length === 0) {
      this.init();
    }
  }
  
  init() {
    this.networks = [];
    
    // Create multiple neural network structures
    for (let i = 0; i < this.networkCount; i++) {
      const layers = Math.floor(Math.random() * 3) + 3; // 3-5 layers
      const nodesPerLayer = [];
      
      // Generate nodes per layer (tapered structure)
      for (let j = 0; j < layers; j++) {
        const progress = j / (layers - 1);
        const minNodes = 3;
        const maxNodes = 8;
        const nodes = Math.floor(minNodes + (maxNodes - minNodes) * (1 - progress * 0.5));
        nodesPerLayer.push(nodes);
      }
      
      this.networks.push({
        x: Math.random() * this.canvas.width,
        y: Math.random() * this.canvas.height,
        vx: (Math.random() - 0.5) * 0.15,
        vy: (Math.random() - 0.5) * 0.15,
        rotation: Math.random() * Math.PI * 2,
        rotationSpeed: (Math.random() - 0.5) * 0.001,
        scale: Math.random() * 0.5 + 0.5, // 0.5 to 1.0 scale (larger)
        layers: layers,
        nodesPerLayer: nodesPerLayer,
        layerSpacing: 100,
        nodeRadius: 4,
        opacity: Math.random() * 0.3 + 0.2, // 0.2 to 0.5 (more visible)
        floatPhase: Math.random() * Math.PI * 2,
        floatAmplitude: Math.random() * 15 + 10,
        pulsePhase: Math.random() * Math.PI * 2,
        pulseSpeed: Math.random() * 0.02 + 0.01
      });
    }
  }
  
  update() {
    this.networks.forEach(network => {
      // Update floating motion
      network.floatPhase += 0.008;
      network.pulsePhase += network.pulseSpeed;
      network.rotation += network.rotationSpeed;
      
      // Smooth floating movement
      network.x += network.vx + Math.sin(network.floatPhase) * 0.05;
      network.y += network.vy + Math.cos(network.floatPhase * 0.7) * 0.05;
      
      // Wrap around edges
      const networkWidth = network.layers * network.layerSpacing * network.scale;
      const networkHeight = Math.max(...network.nodesPerLayer) * 30 * network.scale;
      
      if (network.x < -networkWidth) network.x = this.canvas.width + networkWidth;
      if (network.x > this.canvas.width + networkWidth) network.x = -networkWidth;
      if (network.y < -networkHeight) network.y = this.canvas.height + networkHeight;
      if (network.y > this.canvas.height + networkHeight) network.y = -networkHeight;
    });
  }
  
  drawNetwork(network) {
    this.ctx.save();
    this.ctx.translate(network.x, network.y);
    this.ctx.rotate(network.rotation);
    this.ctx.scale(network.scale, network.scale);
    
    // Calculate network dimensions
    const layerSpacing = network.layerSpacing;
    const maxNodes = Math.max(...network.nodesPerLayer);
    const nodeVerticalSpacing = 35;
    
    // Draw connections first (behind nodes) - more visible
    this.ctx.globalAlpha = network.opacity * 0.5;
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    this.ctx.lineWidth = 1;
    
    for (let layerIdx = 0; layerIdx < network.layers - 1; layerIdx++) {
      const currentLayerNodes = network.nodesPerLayer[layerIdx];
      const nextLayerNodes = network.nodesPerLayer[layerIdx + 1];
      
      const currentLayerX = layerIdx * layerSpacing;
      const nextLayerX = (layerIdx + 1) * layerSpacing;
      
      // Draw connections between layers
      for (let i = 0; i < currentLayerNodes; i++) {
        const currentY = (i - (currentLayerNodes - 1) / 2) * nodeVerticalSpacing;
        
        for (let j = 0; j < nextLayerNodes; j++) {
          const nextY = (j - (nextLayerNodes - 1) / 2) * nodeVerticalSpacing;
          
          // Draw connection
          this.ctx.beginPath();
          this.ctx.moveTo(currentLayerX, currentY);
          this.ctx.lineTo(nextLayerX, nextY);
          this.ctx.stroke();
        }
      }
    }
    
    // Draw nodes - more visible
    const pulse = Math.sin(network.pulsePhase) * 0.2 + 1; // 0.8 to 1.2
    
    for (let layerIdx = 0; layerIdx < network.layers; layerIdx++) {
      const nodesInLayer = network.nodesPerLayer[layerIdx];
      const layerX = layerIdx * layerSpacing;
      
      for (let nodeIdx = 0; nodeIdx < nodesInLayer; nodeIdx++) {
        const nodeY = (nodeIdx - (nodesInLayer - 1) / 2) * nodeVerticalSpacing;
        const radius = network.nodeRadius * pulse;
        
        // Outer glow - more visible
        const gradient = this.ctx.createRadialGradient(
          layerX, nodeY, 0,
          layerX, nodeY, radius * 5
        );
        gradient.addColorStop(0, `rgba(255, 255, 255, ${network.opacity * 0.6})`);
        gradient.addColorStop(0.5, `rgba(255, 255, 255, ${network.opacity * 0.2})`);
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
        
        this.ctx.globalAlpha = network.opacity;
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(layerX, nodeY, radius * 5, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Core node - brighter
        this.ctx.globalAlpha = network.opacity * 2;
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        this.ctx.beginPath();
        this.ctx.arc(layerX, nodeY, radius, 0, Math.PI * 2);
        this.ctx.fill();
      }
    }
    
    this.ctx.restore();
  }
  
  draw() {
    // Clear with very subtle fade for trailing effect
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Draw all networks
    this.networks.forEach(network => this.drawNetwork(network));
  }
  
  animate() {
    this.update();
    this.draw();
    this.animationId = requestAnimationFrame(() => this.animate());
  }
  
  destroy() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('neural-network-canvas');
  if (canvas) {
    try {
      window.floatingNeuralNetworks = new FloatingNeuralNetworks(canvas);
      console.log('Neural networks initialized');
    } catch (error) {
      console.error('Error initializing neural networks:', error);
    }
  } else {
    console.error('Canvas element not found');
  }
});
