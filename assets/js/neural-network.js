// Floating Background Imagery - Subtle and Refined
class FloatingBackground {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.shapes = [];
    this.shapeCount = 15;
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
    this.shapes = [];
    
    // Create floating geometric shapes
    for (let i = 0; i < this.shapeCount; i++) {
      const size = Math.random() * 200 + 100;
      const x = Math.random() * this.canvas.width;
      const y = Math.random() * this.canvas.height;
      
      this.shapes.push({
        x: x,
        y: y,
        size: size,
        rotation: Math.random() * Math.PI * 2,
        rotationSpeed: (Math.random() - 0.5) * 0.002,
        vx: (Math.random() - 0.5) * 0.1,
        vy: (Math.random() - 0.5) * 0.1,
        opacity: Math.random() * 0.15 + 0.05,
        type: Math.floor(Math.random() * 3), // 0: circle, 1: square, 2: triangle
        floatPhase: Math.random() * Math.PI * 2,
        floatAmplitude: Math.random() * 20 + 10
      });
    }
  }
  
  update() {
    this.shapes.forEach(shape => {
      // Update position with floating motion
      shape.floatPhase += 0.01;
      shape.x += shape.vx + Math.sin(shape.floatPhase) * 0.1;
      shape.y += shape.vy + Math.cos(shape.floatPhase) * 0.1;
      shape.rotation += shape.rotationSpeed;
      
      // Wrap around edges
      if (shape.x < -shape.size) shape.x = this.canvas.width + shape.size;
      if (shape.x > this.canvas.width + shape.size) shape.x = -shape.size;
      if (shape.y < -shape.size) shape.y = this.canvas.height + shape.size;
      if (shape.y > this.canvas.height + shape.size) shape.y = -shape.size;
    });
  }
  
  drawShape(shape) {
    this.ctx.save();
    this.ctx.translate(shape.x, shape.y);
    this.ctx.rotate(shape.rotation);
    this.ctx.globalAlpha = shape.opacity;
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    this.ctx.lineWidth = 1;
    this.ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
    
    const halfSize = shape.size / 2;
    
    switch(shape.type) {
      case 0: // Circle
        this.ctx.beginPath();
        this.ctx.arc(0, 0, halfSize, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.stroke();
        break;
        
      case 1: // Square
        this.ctx.beginPath();
        this.ctx.rect(-halfSize, -halfSize, shape.size, shape.size);
        this.ctx.fill();
        this.ctx.stroke();
        break;
        
      case 2: // Triangle
        this.ctx.beginPath();
        this.ctx.moveTo(0, -halfSize);
        this.ctx.lineTo(-halfSize * 0.866, halfSize * 0.5);
        this.ctx.lineTo(halfSize * 0.866, halfSize * 0.5);
        this.ctx.closePath();
        this.ctx.fill();
        this.ctx.stroke();
        break;
    }
    
    this.ctx.restore();
  }
  
  draw() {
    // Clear with very subtle fade for trailing effect
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Draw all shapes
    this.shapes.forEach(shape => this.drawShape(shape));
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
    window.floatingBackground = new FloatingBackground(canvas);
  }
});
