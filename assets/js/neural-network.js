// Animated Neural Network Background
class NeuralNetwork {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.nodes = [];
    this.connections = [];
    this.nodeCount = 50;
    this.connectionDistance = 150;
    this.animationId = null;
    
    this.resize();
    this.init();
    this.animate();
    
    window.addEventListener('resize', () => this.resize());
  }
  
  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }
  
  init() {
    this.nodes = [];
    this.connections = [];
    
    // Create nodes
    for (let i = 0; i < this.nodeCount; i++) {
      this.nodes.push({
        x: Math.random() * this.canvas.width,
        y: Math.random() * this.canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        radius: Math.random() * 2 + 1,
        pulse: Math.random() * Math.PI * 2
      });
    }
    
    // Create connections
    for (let i = 0; i < this.nodes.length; i++) {
      for (let j = i + 1; j < this.nodes.length; j++) {
        const dx = this.nodes[i].x - this.nodes[j].x;
        const dy = this.nodes[i].y - this.nodes[j].y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < this.connectionDistance) {
          this.connections.push({
            node1: i,
            node2: j,
            distance: distance,
            opacity: 1 - distance / this.connectionDistance
          });
        }
      }
    }
  }
  
  update() {
    // Update node positions
    this.nodes.forEach(node => {
      node.x += node.vx;
      node.y += node.vy;
      node.pulse += 0.02;
      
      // Bounce off edges
      if (node.x < 0 || node.x > this.canvas.width) node.vx *= -1;
      if (node.y < 0 || node.y > this.canvas.height) node.vy *= -1;
      
      // Keep in bounds
      node.x = Math.max(0, Math.min(this.canvas.width, node.x));
      node.y = Math.max(0, Math.min(this.canvas.height, node.y));
    });
    
    // Recalculate connections
    this.connections = [];
    for (let i = 0; i < this.nodes.length; i++) {
      for (let j = i + 1; j < this.nodes.length; j++) {
        const dx = this.nodes[i].x - this.nodes[j].x;
        const dy = this.nodes[i].y - this.nodes[j].y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < this.connectionDistance) {
          this.connections.push({
            node1: i,
            node2: j,
            distance: distance,
            opacity: 1 - distance / this.connectionDistance
          });
        }
      }
    }
  }
  
  draw() {
    // Clear with slight fade for trail effect
    this.ctx.fillStyle = 'rgba(10, 10, 15, 0.1)';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Draw connections
    this.connections.forEach(conn => {
      const node1 = this.nodes[conn.node1];
      const node2 = this.nodes[conn.node2];
      
      const gradient = this.ctx.createLinearGradient(
        node1.x, node1.y, node2.x, node2.y
      );
      gradient.addColorStop(0, `rgba(100, 200, 255, ${conn.opacity * 0.3})`);
      gradient.addColorStop(1, `rgba(150, 100, 255, ${conn.opacity * 0.3})`);
      
      this.ctx.strokeStyle = gradient;
      this.ctx.lineWidth = 0.5;
      this.ctx.beginPath();
      this.ctx.moveTo(node1.x, node1.y);
      this.ctx.lineTo(node2.x, node2.y);
      this.ctx.stroke();
    });
    
    // Draw nodes
    this.nodes.forEach(node => {
      const pulseSize = Math.sin(node.pulse) * 0.5 + 1;
      const radius = node.radius * pulseSize;
      
      // Outer glow
      const gradient = this.ctx.createRadialGradient(
        node.x, node.y, 0,
        node.x, node.y, radius * 3
      );
      gradient.addColorStop(0, 'rgba(100, 200, 255, 0.8)');
      gradient.addColorStop(0.5, 'rgba(150, 100, 255, 0.4)');
      gradient.addColorStop(1, 'rgba(100, 200, 255, 0)');
      
      this.ctx.fillStyle = gradient;
      this.ctx.beginPath();
      this.ctx.arc(node.x, node.y, radius * 3, 0, Math.PI * 2);
      this.ctx.fill();
      
      // Core node
      this.ctx.fillStyle = 'rgba(150, 200, 255, 0.9)';
      this.ctx.beginPath();
      this.ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
      this.ctx.fill();
    });
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
    window.neuralNetwork = new NeuralNetwork(canvas);
  }
});

