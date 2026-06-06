/**
 * CVCRAFT Web UI — Three.js 3D Scene Viewer + API Client
 */

// ---------------------------------------------------------------------------
// Dependency check
// ---------------------------------------------------------------------------
if (typeof THREE === 'undefined') {
  document.getElementById('viewport-container').innerHTML =
    '<p style="padding:2rem;color:#e94560;">Failed to load Three.js from CDN. Check your internet connection or firewall settings.</p>';
  throw new Error('Three.js not loaded');
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let currentScene = null;
let selectedBlocks = new Set();
let blockMeshes = {};
let collapsed3DStages = {};  // Track which stages are collapsed in the 3D viewport

// ---------------------------------------------------------------------------
// Three.js Setup
// ---------------------------------------------------------------------------
const canvas = document.getElementById('viewport');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);

const scene3d = new THREE.Scene();
scene3d.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 500000);
camera.position.set(30, 20, 30);
camera.lookAt(0, 0, 0);

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
scene3d.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(20, 30, 20);
scene3d.add(dirLight);

// Grid helper (hidden by default to avoid obstructing the view)
const grid = new THREE.GridHelper(60, 60, 0x222233, 0x1a1a2e);
grid.material.transparent = true;
grid.material.opacity = 0.15;
grid.visible = false;
scene3d.add(grid);

// Stage colors (fallback)
const STAGE_COLORS = {
  input: 0x9CA3AF,
  backbone: 0x2563EB,
  neck: 0xD97706,
  head: 0xDC2626,
  output: 0x059669,
};

// Operator-type colors: color blocks by what they DO
const OP_TYPE_COLORS = {
  // Convolutions (blue family)
  'Conv2d': 0x3B82F6,
  'ConvBlock': 0x2563EB,
  'DWConvBlock': 0x1D4ED8,
  'PWConvBlock': 0x1E40AF,
  'MBConv': 0x1E3A8A,
  'GhostBlock': 0x60A5FA,
  'DepthwiseConv': 0x93C5FD,
  // Normalization (teal/cyan)
  'BatchNorm2d': 0x06B6D4,
  'BatchNormBlock': 0x06B6D4,
  'GroupNormBlock': 0x0891B2,
  'SyncBNBlock': 0x0E7490,
  // Activation (green)
  'ReLU': 0x10B981,
  'ReLUBlock': 0x10B981,
  'SiLU': 0x059669,
  'SiLUBlock': 0x059669,
  'HSwishBlock': 0x047857,
  'Sigmoid': 0x34D399,
  'SigmoidBlock': 0x34D399,
  // Pooling (purple)
  'MaxPool2d': 0x8B5CF6,
  'AvgPool2d': 0x7C3AED,
  'PoolingBlock': 0x6D28D9,
  'SPPBlock': 0xA78BFA,
  'SPPFBlock': 0xA78BFA,
  // Upsample / resize (pink)
  'Upsample': 0xEC4899,
  'UpsampleBlock': 0xEC4899,
  'ResizeFeatureMap': 0xF472B6,
  'Downsample': 0xDB2777,
  // Structural / merge ops (amber/orange)
  'Concat': 0xF59E0B,
  'ConcatBlock': 0xF59E0B,
  'Add': 0xD97706,
  'AddBlock': 0xD97706,
  'Split': 0xFBBF24,
  'Merge': 0xFBBF24,
  // Residual / composite (indigo)
  'ResidualBlock': 0x6366F1,
  'CSPBlock': 0x4F46E5,
  'FocusBlock': 0x4338CA,
  // Detection heads (red family)
  'DetectHead': 0xEF4444,
  'DecoupledHeadBlock': 0xDC2626,
  'ClsHeadBlock': 0xF87171,
  'RegHeadBlock': 0xB91C1C,
  'CenterHeadBlock': 0xFCA5A5,
  'DFLBlock': 0x991B1B,
  'NMSFreeDecodeBlock': 0x7F1D1D,
  // Neck (orange)
  'FPNBlock': 0xEA580C,
  'PANBlock': 0xC2410C,
  'BiFPNBlock': 0xFB923C,
  // I/O (gray)
  'Input': 0x9CA3AF,
  'InputBlock': 0x9CA3AF,
  'Output': 0x6B7280,
  'OutputBlock': 0x6B7280,
  // Utility (slate)
  'Dropout': 0x64748B,
  'DropPathBlock': 0x64748B,
  'Linear': 0x475569,
  'Flatten': 0x334155,
  'Reshape': 0x475569,
  'Transpose': 0x475569,
  'Permute': 0x475569,
  'Identity': 0x94A3B8,
  'IdentityBlock': 0x94A3B8,
};

// Operator category for legend
const OP_CATEGORIES = {
  'Convolution': 0x2563EB,
  'Normalization': 0x06B6D4,
  'Activation': 0x10B981,
  'Pooling': 0x8B5CF6,
  'Upsample': 0xEC4899,
  'Merge/Split': 0xF59E0B,
  'Composite': 0x6366F1,
  'Detection': 0xEF4444,
  'Neck': 0xEA580C,
  'I/O': 0x9CA3AF,
};

function getBlockColor(block) {
  const type = block.type;
  if (OP_TYPE_COLORS[type] !== undefined) return OP_TYPE_COLORS[type];
  // Fallback to stage color
  const stageId = (block.meta && block.meta.stage_id) || 'backbone';
  return STAGE_COLORS[stageId] || 0x2563EB;
}

const FROZEN_OPACITY = 0.4;
const NON_RELEVANT_OPACITY = 0.3;

// ---------------------------------------------------------------------------
// Orbit Controls
// Right-click: Rotate | Middle-click: Pan | Scroll: Zoom | Left-click: Select
// ---------------------------------------------------------------------------
let isRotating = false;
let isPanning = false;
let dragMoved = false;
let prevMouse = { x: 0, y: 0 };
let spherical = { theta: Math.PI / 4, phi: Math.PI / 4, radius: 50 };
let panOffset = { x: 0, y: 0, z: 0 };

function updateCameraFromSpherical() {
  camera.position.x = panOffset.x + spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta);
  camera.position.y = panOffset.y + spherical.radius * Math.cos(spherical.phi);
  camera.position.z = panOffset.z + spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta);
  camera.lookAt(panOffset.x, panOffset.y, panOffset.z);
}

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

canvas.addEventListener('mousedown', (e) => {
  // Right mouse button (button === 2) for rotation
  if (e.button === 2) {
    e.preventDefault();
    isRotating = true;
    dragMoved = false;
    prevMouse = { x: e.clientX, y: e.clientY };
  }
  // Middle mouse button (button === 1) for panning
  else if (e.button === 1) {
    e.preventDefault();
    isPanning = true;
    dragMoved = false;
    prevMouse = { x: e.clientX, y: e.clientY };
  }
  // Left-click (button === 0): reset dragMoved so selection works after a drag
  else if (e.button === 0) {
    dragMoved = false;
  }
});

canvas.addEventListener('mousemove', (e) => {
  if (isPanning) {
    dragMoved = true;
    const dx = e.clientX - prevMouse.x;
    const dy = e.clientY - prevMouse.y;
    const panSpeed = spherical.radius * 0.002;
    const right = new THREE.Vector3();
    const up = new THREE.Vector3();
    right.setFromMatrixColumn(camera.matrixWorld, 0);
    up.setFromMatrixColumn(camera.matrixWorld, 1);
    panOffset.x -= right.x * dx * panSpeed - up.x * dy * panSpeed;
    panOffset.y -= right.y * dx * panSpeed - up.y * dy * panSpeed;
    panOffset.z -= right.z * dx * panSpeed - up.z * dy * panSpeed;
    updateCameraFromSpherical();
    prevMouse = { x: e.clientX, y: e.clientY };
    return;
  }
  if (isRotating) {
    dragMoved = true;
    const dx = e.clientX - prevMouse.x;
    const dy = e.clientY - prevMouse.y;
    spherical.theta += dx * 0.005;
    spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi + dy * 0.005));
    updateCameraFromSpherical();
    prevMouse = { x: e.clientX, y: e.clientY };
  }
});

canvas.addEventListener('mouseup', () => { isRotating = false; isPanning = false; });
canvas.addEventListener('mouseleave', () => { isRotating = false; isPanning = false; });

// Prevent context menu on middle click
canvas.addEventListener('auxclick', (e) => { if (e.button === 1) e.preventDefault(); });

// Nearly unlimited zoom: min 0.1, max 100000
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const zoomFactor = 1 + e.deltaY * 0.001;
  spherical.radius = Math.max(0.1, Math.min(100000, spherical.radius * zoomFactor));
  updateCameraFromSpherical();
}, { passive: false });

updateCameraFromSpherical();

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------
function resize() {
  const container = document.getElementById('viewport-container');
  const w = container.clientWidth;
  const h = container.clientHeight;
  if (w === 0 || h === 0) return;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
window.addEventListener('load', resize);
requestAnimationFrame(resize);

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------
function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene3d, camera);
}
animate();

// ---------------------------------------------------------------------------
// Scene rendering
// ---------------------------------------------------------------------------
function clearScene3D() {
  Object.values(blockMeshes).forEach(mesh => scene3d.remove(mesh));
  blockMeshes = {};
  // Remove edges
  scene3d.children.filter(c => c.userData && c.userData.isEdge).forEach(c => scene3d.remove(c));
}

function renderScene(sceneData) {
  clearScene3D();
  if (!sceneData || !sceneData.blocks) return;

  const blocks = sceneData.blocks;
  const edges = sceneData.edges || [];

  // Track collapsed groups for compact mode
  const stageGroups = {};
  blocks.forEach(block => {
    const stageId = (block.meta && block.meta.stage_id) || 'backbone';
    if (!stageGroups[stageId]) stageGroups[stageId] = [];
    stageGroups[stageId].push(block);
  });

  blocks.forEach(block => {
    const frozen = block.meta && block.meta.frozen;
    const anchorFreeRelevant = block.meta && block.meta.anchor_free_relevant !== false;
    let color = getBlockColor(block);
    
    // Desaturate color for non-relevant blocks
    if (!anchorFreeRelevant) {
      const r = (color >> 16) & 0xff;
      const g = (color >> 8) & 0xff;
      const b = color & 0xff;
      const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      const mixR = Math.round(gray * 0.7 + r * 0.3);
      const mixG = Math.round(gray * 0.7 + g * 0.3);
      const mixB = Math.round(gray * 0.7 + b * 0.3);
      color = (mixR << 16) | (mixG << 8) | mixB;
    }

    // Compact block: smaller height, wider to look like a chip/node
    const geometry = new THREE.BoxGeometry(1.8, 0.6, 1.0);
    const material = new THREE.MeshPhongMaterial({
      color,
      transparent: frozen || !anchorFreeRelevant,
      opacity: !anchorFreeRelevant ? NON_RELEVANT_OPACITY : (frozen ? FROZEN_OPACITY : 1.0),
    });
    const mesh = new THREE.Mesh(geometry, material);

    // Add subtle edge wireframe for definition
    const edgeGeo = new THREE.EdgesGeometry(geometry);
    const edgeMat = new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.15, transparent: true });
    const wireframe = new THREE.LineSegments(edgeGeo, edgeMat);
    mesh.add(wireframe);

    const pos = block.position || { x: 0, y: 0, z: 0 };
    mesh.position.set(pos.x, pos.y, pos.z);
    mesh.userData = { blockId: block.id, block };
    scene3d.add(mesh);
    blockMeshes[block.id] = mesh;
  });

  // Draw edges as Netron-style connections (straight vertical or smooth S-curves)
  const blockById = {};
  blocks.forEach(b => { blockById[b.id] = b; });

  edges.forEach(edge => {
    const fromBlock = blockById[edge.from];
    const toBlock = blockById[edge.to];
    if (!fromBlock || !toBlock) return;
    const from = fromBlock.position || { x: 0, y: 0, z: 0 };
    const to = toBlock.position || { x: 0, y: 0, z: 0 };

    const startY = from.y - 0.35;
    const endY = to.y + 0.35;
    let points;

    if (Math.abs(from.x - to.x) < 0.01 && Math.abs(from.z - to.z) < 0.01) {
      // Same column: straight vertical line (Netron-style)
      points = [
        new THREE.Vector3(from.x, startY, from.z),
        new THREE.Vector3(to.x, endY, to.z),
      ];
    } else {
      // Different columns: smooth S-curve (Netron-style bezier)
      const midY = (startY + endY) / 2;
      points = [
        new THREE.Vector3(from.x, startY, from.z),
        new THREE.Vector3(from.x, midY, from.z),
        new THREE.Vector3(to.x, midY, to.z),
        new THREE.Vector3(to.x, endY, to.z),
      ];
    }

    let geometry;
    if (points.length === 2) {
      geometry = new THREE.BufferGeometry().setFromPoints(points);
    } else {
      const curve = new THREE.CatmullRomCurve3(points);
      geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(24));
    }
    const material = new THREE.LineBasicMaterial({ color: 0x8899AA, linewidth: 1, opacity: 0.8, transparent: true });
    const line = new THREE.Line(geometry, material);
    line.userData = { isEdge: true, edgeFrom: edge.from, edgeTo: edge.to };
    scene3d.add(line);

    // Arrow head at target end (small triangle pointing down)
    const arrowSize = 0.15;
    const arrowY = endY + 0.05;
    const arrowGeo = new THREE.BufferGeometry();
    const arrowVerts = new Float32Array([
      to.x, arrowY + arrowSize, to.z,
      to.x - arrowSize * 0.5, arrowY + arrowSize * 2, to.z,
      to.x + arrowSize * 0.5, arrowY + arrowSize * 2, to.z,
    ]);
    arrowGeo.setAttribute('position', new THREE.BufferAttribute(arrowVerts, 3));
    const arrowMat = new THREE.MeshBasicMaterial({ color: 0x8899AA, opacity: 0.8, transparent: true, side: THREE.DoubleSide });
    const arrowMesh = new THREE.Mesh(arrowGeo, arrowMat);
    arrowMesh.userData = { isEdge: true, edgeFrom: edge.from, edgeTo: edge.to };
    scene3d.add(arrowMesh);
  });
}

// ---------------------------------------------------------------------------
// Selection via raycasting
// ---------------------------------------------------------------------------
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

canvas.addEventListener('click', (e) => {
  if (dragMoved) return;
  const rect = canvas.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);

  const meshes = Object.values(blockMeshes);
  // Include children (wireframes) in intersection, then resolve to parent block
  const intersects = raycaster.intersectObjects(meshes, true);

  if (intersects.length > 0) {
    // Traverse up to find the mesh with blockId (may have hit wireframe child)
    let hitObj = intersects[0].object;
    while (hitObj && !hitObj.userData.blockId) {
      hitObj = hitObj.parent;
    }
    if (!hitObj || !hitObj.userData.blockId) {
      selectedBlocks.clear();
      updateSelectionVisuals();
      updateSelectionInfo();
      updatePropertyEditor();
      return;
    }
    const blockId = hitObj.userData.blockId;

    if (e.ctrlKey || e.metaKey) {
      // Multi-select toggle
      if (selectedBlocks.has(blockId)) {
        selectedBlocks.delete(blockId);
      } else {
        selectedBlocks.add(blockId);
      }
    } else {
      selectedBlocks.clear();
      selectedBlocks.add(blockId);
    }
  } else {
    selectedBlocks.clear();
  }
  updateSelectionVisuals();
  updateSelectionInfo();
  updatePropertyEditor();
});

function updateSelectionVisuals() {
  Object.entries(blockMeshes).forEach(([id, mesh]) => {
    const block = mesh.userData.block;
    const frozen = block && block.meta && block.meta.frozen;
    if (selectedBlocks.has(id)) {
      mesh.material.emissive = new THREE.Color(0xe94560);
      mesh.material.emissiveIntensity = 0.5;
    } else {
      mesh.material.emissive = new THREE.Color(0x000000);
      mesh.material.emissiveIntensity = 0;
    }
  });
}

function updateSelectionInfo() {
  const info = document.getElementById('selection-info');
  if (selectedBlocks.size === 0) {
    info.innerHTML = '<em>No block selected</em>';
    return;
  }
  const lines = [];
  let totalParams = 0;
  selectedBlocks.forEach(id => {
    const mesh = blockMeshes[id];
    if (!mesh) return;
    const b = mesh.userData.block;
    const params = b.params || b.meta?.parameters || 0;
    totalParams += params;
    const paramsStr = params >= 1e6 ? (params / 1e6).toFixed(2) + 'M' : params >= 1e3 ? (params / 1e3).toFixed(1) + 'K' : params.toLocaleString();
    
    const anchorFreeRelevant = b.meta && b.meta.anchor_free_relevant !== false;
    const relevanceIcon = anchorFreeRelevant ? '✓' : '⚠';
    const relevanceLabel = anchorFreeRelevant ? 'Relevant' : 'Non-relevant';

    lines.push(`━━━━━━━━━━━━━━━━━━━━━━━━`);
    lines.push(`🔷 Block: ${b.id}`);
    lines.push(`   Type: ${b.type}`);
    lines.push(`   Stage: ${b.meta?.stage_id || '?'}`);
    lines.push(`   Frozen: ${b.meta?.frozen ? '❄ Yes' : '🔥 No'}`);
    lines.push(`   Anchor-free: ${relevanceIcon} ${relevanceLabel}`);
    lines.push(`   Parameters: ${paramsStr}`);
    lines.push(``);
    // Pedagogical details
    lines.push(`   📚 Description:`);
    lines.push(`   ${getBlockDescription(b.type)}`);
    lines.push(``);
    if (!anchorFreeRelevant) {
      lines.push(`   ⚠️  Note:`);
      lines.push(`   This operation is not typically used in`);
      lines.push(`   anchor-free object detection models.`);
      lines.push(``);
    }
    if (b.meta?.in_channels || b.meta?.out_channels) {
      lines.push(`   📐 Dimensions:`);
      if (b.meta?.in_channels) lines.push(`     In channels: ${b.meta.in_channels}`);
      if (b.meta?.out_channels) lines.push(`     Out channels: ${b.meta.out_channels}`);
      if (b.meta?.kernel_size) lines.push(`     Kernel: ${b.meta.kernel_size}`);
      if (b.meta?.stride) lines.push(`     Stride: ${b.meta.stride}`);
      lines.push(``);
    }
    if (b.meta?.resolution_level !== undefined) {
      lines.push(`   🎚 Resolution level: ${b.meta.resolution_level}`);
    }
  });

  if (selectedBlocks.size > 1) {
    const totalStr = totalParams >= 1e6 ? (totalParams / 1e6).toFixed(2) + 'M' : totalParams >= 1e3 ? (totalParams / 1e3).toFixed(1) + 'K' : totalParams.toLocaleString();
    lines.push(`━━━━━━━━━━━━━━━━━━━━━━━━`);
    lines.push(`📊 Total selected: ${selectedBlocks.size} blocks`);
    lines.push(`📊 Total params: ${totalStr}`);
  }

  info.textContent = lines.join('\n');
}

// Pedagogical descriptions of common block types
function getBlockDescription(type) {
  const descriptions = {
    'Conv2d': 'Applies a 2D convolution over input. Extracts spatial features using learnable filters.',
    'ConvBlock': 'Convolution + Normalization + Activation. Standard building block for feature extraction.',
    'DWConvBlock': 'Depthwise separable convolution. Reduces computation by factoring spatial and channel mixing.',
    'BatchNorm2d': 'Normalizes activations per batch. Stabilizes training and allows higher learning rates.',
    'ReLU': 'Rectified Linear Unit. Introduces non-linearity: f(x) = max(0, x).',
    'SiLU': 'Sigmoid Linear Unit (Swish). Smooth non-linearity: f(x) = x * σ(x).',
    'MaxPool2d': 'Downsamples by taking max value in each window. Reduces spatial dimensions.',
    'AvgPool2d': 'Downsamples by averaging values in each window. Smoother than max pooling.',
    'Linear': 'Fully connected layer. Learns a linear transformation of the input features.',
    'Dropout': 'Randomly zeros elements during training. Regularization technique to prevent overfitting.',
    'Concat': 'Concatenates multiple feature maps along channel dimension. Merges multi-scale features.',
    'Add': 'Element-wise addition of feature maps. Used in residual/skip connections.',
    'Upsample': 'Increases spatial resolution. Used in decoder/neck to recover fine-grained details.',
    'CSPBlock': 'Cross-Stage Partial block. Splits features, processes one half, and merges back efficiently.',
    'SPPFBlock': 'Spatial Pyramid Pooling (Fast). Captures multi-scale context via pooling at different sizes.',
    'FocusBlock': 'Slices input into 4 parts and concatenates. Reduces spatial size without info loss.',
    'DetectHead': 'Detection head. Produces bounding box predictions, class scores, and objectness.',
    'Input': 'Model input tensor. Entry point of the neural network receiving raw data.',
    'Output': 'Model output tensor. Final predictions produced by the network.',
  };
  return descriptions[type] || `Neural network operation of type "${type}". Transforms input features in the pipeline.`;
}

// ---------------------------------------------------------------------------
// Netron-style Property Editor (editable fields per selected block)
// ---------------------------------------------------------------------------
function updatePropertyEditor() {
  const container = document.getElementById('property-editor');
  if (!container) return;

  if (selectedBlocks.size !== 1 || !currentScene) {
    container.innerHTML = '<em>Select a single block to edit properties</em>';
    return;
  }

  const blockId = [...selectedBlocks][0];
  const block = currentScene.blocks.find(b => b.id === blockId);
  if (!block) {
    container.innerHTML = '<em>Block not found</em>';
    return;
  }

  let html = `<div class="prop-editor-title">${escapeHtml(block.id)} <span class="prop-type">[${escapeHtml(block.type)}]</span></div>`;

  // Type field
  html += `<div class="prop-section">`;
  html += `<div class="prop-section-header" data-section="type">▼ Type</div>`;
  html += `<div class="prop-section-body">`;
  html += `<div class="prop-row"><label>type</label><input class="prop-input" data-category="type" data-key="type" value="${escapeHtml(block.type)}"></div>`;
  html += `</div></div>`;

  // Params section (collapsible)
  const params = block.params || {};
  html += `<div class="prop-section">`;
  html += `<div class="prop-section-header" data-section="params">▼ Parameters</div>`;
  html += `<div class="prop-section-body">`;
  const paramKeys = Object.keys(params);
  if (paramKeys.length === 0) {
    html += `<div class="prop-row"><em>No parameters</em></div>`;
  } else {
    paramKeys.forEach(key => {
      const val = params[key];
      const displayVal = typeof val === 'object' ? JSON.stringify(val) : String(val);
      html += `<div class="prop-row"><label>${escapeHtml(key)}</label><input class="prop-input" data-category="params" data-key="${escapeHtml(key)}" value="${escapeHtml(displayVal)}"></div>`;
    });
  }
  html += `</div></div>`;

  // Meta section (collapsible)
  const meta = block.meta || {};
  html += `<div class="prop-section">`;
  html += `<div class="prop-section-header" data-section="meta">▼ Metadata</div>`;
  html += `<div class="prop-section-body">`;
  const metaKeys = Object.keys(meta);
  if (metaKeys.length === 0) {
    html += `<div class="prop-row"><em>No metadata</em></div>`;
  } else {
    metaKeys.forEach(key => {
      const val = meta[key];
      const displayVal = typeof val === 'object' ? JSON.stringify(val) : String(val);
      const readonly = (key === 'topo_index') ? ' readonly title="Auto-computed"' : '';
      html += `<div class="prop-row"><label>${escapeHtml(key)}</label><input class="prop-input" data-category="meta" data-key="${escapeHtml(key)}" value="${escapeHtml(displayVal)}"${readonly}></div>`;
    });
  }
  html += `</div></div>`;

  // IO section (read-only)
  const io = block.io || {};
  html += `<div class="prop-section">`;
  html += `<div class="prop-section-header" data-section="io">▼ I/O Tensors</div>`;
  html += `<div class="prop-section-body">`;
  html += `<div class="prop-row"><label>in</label><input class="prop-input" value="${escapeHtml(JSON.stringify(io.in || []))}" readonly title="Read-only"></div>`;
  html += `<div class="prop-row"><label>out</label><input class="prop-input" value="${escapeHtml(JSON.stringify(io.out || []))}" readonly title="Read-only"></div>`;
  html += `</div></div>`;

  // Apply button
  html += `<button class="prop-apply-btn" id="prop-apply">✔ Apply Changes</button>`;
  html += `<button class="prop-apply-btn" id="prop-fill-defaults" style="margin-left:0.5rem;background:#475569;" title="Fill in recommended hyperparameters for this block type">💡 Fill Defaults</button>`;
  html += `<div id="prop-status" class="prop-status"></div>`;

  container.innerHTML = html;

  // Collapsible sections within property editor
  container.querySelectorAll('.prop-section-header').forEach(header => {
    header.addEventListener('click', () => {
      const body = header.nextElementSibling;
      if (body.classList.contains('hidden')) {
        body.classList.remove('hidden');
        header.textContent = '▼ ' + header.textContent.substring(2);
      } else {
        body.classList.add('hidden');
        header.textContent = '▶ ' + header.textContent.substring(2);
      }
    });
  });

  // Apply button handler
  document.getElementById('prop-apply').addEventListener('click', () => applyPropertyEdits(blockId));
  document.getElementById('prop-fill-defaults').addEventListener('click', () => fillBlockDefaults(blockId));
}

async function fillBlockDefaults(blockId) {
  const block = currentScene.blocks.find(b => b.id === blockId);
  if (!block) return;
  const statusEl = document.getElementById('prop-status');
  try {
    let defaults;
    if (blockDefaultsCatalog && blockDefaultsCatalog[block.type]) {
      defaults = blockDefaultsCatalog[block.type];
    } else {
      const res = await fetch(`/api/templates/block-defaults/${encodeURIComponent(block.type)}`);
      const data = await res.json();
      defaults = data.params || {};
    }
    if (!defaults || Object.keys(defaults).length === 0) {
      statusEl.textContent = `ℹ No default hyperparameters known for ${block.type}`;
      statusEl.className = 'prop-status';
      return;
    }
    const params = Object.assign({}, defaults, block.params || {});
    const data = await apiPost('/api/edit/update-block', {
      scene: currentScene,
      id: blockId,
      updates: { params },
    });
    loadScene(data.scene);
    showToast(`Filled defaults for ${block.type}`);
  } catch (e) {
    statusEl.textContent = `✘ ${e.message}`;
    statusEl.className = 'prop-status error';
    showToast(`Fill defaults failed: ${e.message}`, 'error');
  }
}

async function applyPropertyEdits(blockId) {
  const container = document.getElementById('property-editor');
  const statusEl = document.getElementById('prop-status');
  const inputs = container.querySelectorAll('.prop-input:not([readonly])');

  const updates = {};
  inputs.forEach(input => {
    const category = input.dataset.category;
    const key = input.dataset.key;
    let value = input.value;

    // Attempt to parse numeric / boolean / JSON values
    if (value === 'true') value = true;
    else if (value === 'false') value = false;
    else if (!isNaN(value) && value.trim() !== '') value = Number(value);
    else {
      try { value = JSON.parse(value); } catch (e) { /* keep as string */ }
    }

    if (category === 'type') {
      updates.type = value;
    } else {
      if (!updates[category]) updates[category] = {};
      updates[category][key] = value;
    }
  });

  statusEl.textContent = '⏳ Validating...';
  statusEl.className = 'prop-status';

  try {
    const data = await apiPost('/api/edit/update-block', {
      scene: currentScene,
      id: blockId,
      updates,
    });
    loadScene(data.scene);
    statusEl.textContent = '✔ Applied successfully';
    statusEl.className = 'prop-status success';
    showToast('Properties updated');
  } catch (e) {
    statusEl.textContent = `✘ ${e.message}`;
    statusEl.className = 'prop-status error';
    showToast(`Rejected: ${e.message}`, 'error');
  }
}

// ---------------------------------------------------------------------------
// 3D Collapsible Stage Groups
// ---------------------------------------------------------------------------
function toggle3DStage(stageId) {
  collapsed3DStages[stageId] = !collapsed3DStages[stageId];
  if (!currentScene || !currentScene.blocks) return;

  currentScene.blocks.forEach(block => {
    const blockStage = (block.meta && block.meta.stage_id) || 'other';
    if (blockStage === stageId) {
      const mesh = blockMeshes[block.id];
      if (mesh) {
        mesh.visible = !collapsed3DStages[stageId];
      }
    }
  });

  // Re-draw edges for visibility
  updateEdgeVisibility();
  update3DStageControls();
}

function updateEdgeVisibility() {
  if (!currentScene) return;
  const blockById = {};
  currentScene.blocks.forEach(b => { blockById[b.id] = b; });

  scene3d.children.forEach(child => {
    if (!child.userData || !child.userData.isEdge) return;
    if (child.userData.edgeFrom && child.userData.edgeTo) {
      const fromBlock = blockById[child.userData.edgeFrom];
      const toBlock = blockById[child.userData.edgeTo];
      const fromStage = fromBlock ? (fromBlock.meta && fromBlock.meta.stage_id) || 'other' : 'other';
      const toStage = toBlock ? (toBlock.meta && toBlock.meta.stage_id) || 'other' : 'other';
      child.visible = !collapsed3DStages[fromStage] && !collapsed3DStages[toStage];
    }
  });
}

function update3DStageControls() {
  const controlsEl = document.getElementById('stage-3d-controls');
  if (!controlsEl || !currentScene || !currentScene.blocks) {
    if (controlsEl) controlsEl.innerHTML = '';
    return;
  }

  const stages = new Set();
  currentScene.blocks.forEach(b => stages.add((b.meta && b.meta.stage_id) || 'other'));

  let html = '';
  const stageOrder = ['input', 'backbone', 'neck', 'head', 'output', 'other'];
  const sorted = [...stages].sort((a, b) => {
    const ia = stageOrder.indexOf(a);
    const ib = stageOrder.indexOf(b);
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
  });

  sorted.forEach(stage => {
    const isCollapsed = collapsed3DStages[stage];
    const icon = isCollapsed ? '▶' : '▼';
    const cls = isCollapsed ? 'stage-ctrl collapsed' : 'stage-ctrl';
    html += `<button class="${cls}" data-stage="${escapeHtml(stage)}" title="${isCollapsed ? 'Show' : 'Hide'} ${stage}">${icon} ${escapeHtml(stage)}</button>`;
  });

  controlsEl.innerHTML = html;
  controlsEl.querySelectorAll('.stage-ctrl').forEach(btn => {
    btn.addEventListener('click', () => toggle3DStage(btn.dataset.stage));
  });
}

// ---------------------------------------------------------------------------
// Metrics display
// ---------------------------------------------------------------------------
function updateMetrics() {
  if (!currentScene || !currentScene.metrics) {
    document.getElementById('metric-flops').textContent = 'FLOPs: —';
    document.getElementById('metric-params').textContent = 'Params: —';
    document.getElementById('metric-latency').textContent = 'Latency: —';
    return;
  }
  const m = currentScene.metrics;
  const flops = m.flops >= 1e9 ? (m.flops / 1e9).toFixed(1) + ' G' : (m.flops / 1e6).toFixed(1) + ' M';
  const params = m.parameters >= 1e6 ? (m.parameters / 1e6).toFixed(2) + ' M' : m.parameters.toLocaleString();
  const latency = m.latencyMs ? Object.entries(m.latencyMs).map(([k, v]) => `${k}: ${v}ms`).join(' | ') : '—';
  document.getElementById('metric-flops').textContent = `FLOPs: ${flops}`;
  document.getElementById('metric-params').textContent = `Params: ${params}`;
  document.getElementById('metric-latency').textContent = `Latency: ${latency}`;
}

function updateSceneInfo() {
  const info = document.getElementById('scene-info');
  if (!currentScene) {
    info.innerHTML = '<em>No scene loaded</em>';
    return;
  }
  const model = currentScene.model || {};
  info.textContent = `Model: ${model.name || '?'}\nFamily: ${model.family || '?'}\nBlocks: ${(currentScene.blocks || []).length}\nEdges: ${(currentScene.edges || []).length}`;
}

// ---------------------------------------------------------------------------
// Block Tree (collapsible stage groups)
// ---------------------------------------------------------------------------
let treeCollapsedStages = {};

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function updateBlockTree() {
  const container = document.getElementById('block-tree');
  if (!currentScene || !currentScene.blocks || currentScene.blocks.length === 0) {
    container.innerHTML = '<em>No scene loaded</em>';
    return;
  }

  // Group blocks by stage
  const stageGroups = {};
  currentScene.blocks.forEach(block => {
    const stageId = (block.meta && block.meta.stage_id) || 'other';
    if (!stageGroups[stageId]) stageGroups[stageId] = [];
    stageGroups[stageId].push(block);
  });

  // All stages collapsed by default if tree is large (>20 blocks)
  const totalBlocks = currentScene.blocks.length;
  if (Object.keys(treeCollapsedStages).length === 0 && totalBlocks > 20) {
    Object.keys(stageGroups).forEach(stage => {
      treeCollapsedStages[stage] = true;
    });
  }

  let html = '';
  const stageOrder = ['input', 'backbone', 'neck', 'head', 'output', 'other'];
  const sortedStages = Object.keys(stageGroups).sort((a, b) => {
    const ia = stageOrder.indexOf(a);
    const ib = stageOrder.indexOf(b);
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
  });

  sortedStages.forEach(stageId => {
    const blocks = stageGroups[stageId];
    const collapsed = treeCollapsedStages[stageId];
    const count = blocks.length;
    const safeStageId = escapeHtml(stageId);
    html += `<div class="tree-stage">`;
    html += `<div class="tree-stage-header ${collapsed ? 'collapsed' : ''}" data-stage="${safeStageId}">`;
    html += `<span class="tree-toggle">▼</span>`;
    html += `<span>${safeStageId} (${count})</span>`;
    html += `</div>`;
    html += `<div class="tree-stage-items ${collapsed ? 'hidden' : ''}">`;
    blocks.forEach(block => {
      const sel = selectedBlocks.has(block.id) ? ' selected' : '';
      const frozen = (block.meta && block.meta.frozen) ? ' ❄' : '';
      const safeId = escapeHtml(block.id);
      const safeType = escapeHtml(block.type);
      html += `<div class="tree-block-item${sel}" data-block-id="${safeId}">${safeId} <span style="opacity:0.6">[${safeType}]</span>${frozen}</div>`;
    });
    html += `</div></div>`;
  });

  container.innerHTML = html;

  // Attach event listeners for collapse/expand
  container.querySelectorAll('.tree-stage-header').forEach(header => {
    header.addEventListener('click', () => {
      const stage = header.dataset.stage;
      treeCollapsedStages[stage] = !treeCollapsedStages[stage];
      header.classList.toggle('collapsed');
      const items = header.nextElementSibling;
      items.classList.toggle('hidden');
    });
  });

  // Attach event listeners for block selection
  container.querySelectorAll('.tree-block-item').forEach(item => {
    item.addEventListener('click', (e) => {
      const blockId = item.dataset.blockId;
      if (e.ctrlKey || e.metaKey) {
        if (selectedBlocks.has(blockId)) {
          selectedBlocks.delete(blockId);
        } else {
          selectedBlocks.add(blockId);
        }
      } else {
        selectedBlocks.clear();
        selectedBlocks.add(blockId);
      }
      updateSelectionVisuals();
      updateSelectionInfo();
      updatePropertyEditor();
      updateBlockTree();
    });
  });
}

function loadScene(sceneData) {
  currentScene = sceneData;
  selectedBlocks.clear();
  treeCollapsedStages = {};
  renderScene(sceneData);
  // Apply 3D stage visibility (set mesh visibility directly without toggling state)
  if (currentScene && currentScene.blocks) {
    currentScene.blocks.forEach(block => {
      const stage = (block.meta && block.meta.stage_id) || 'other';
      if (collapsed3DStages[stage]) {
        const mesh = blockMeshes[block.id];
        if (mesh) mesh.visible = false;
      }
    });
    updateEdgeVisibility();
  }
  updateMetrics();
  updateSceneInfo();
  updateSelectionInfo();
  updateBlockTree();
  updatePropertyEditor();
  update3DStageControls();
}

// ---------------------------------------------------------------------------
// Toast notifications
// ---------------------------------------------------------------------------
function showToast(message, type = 'success') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3200);
}

// ---------------------------------------------------------------------------
// Modal helpers
// ---------------------------------------------------------------------------
function showModal(html) {
  document.getElementById('modal-body').innerHTML = html;
  document.getElementById('modal-overlay').classList.remove('hidden');
}

function hideModal() {
  document.getElementById('modal-overlay').classList.add('hidden');
}

document.getElementById('modal-close').addEventListener('click', hideModal);
document.getElementById('modal-overlay').addEventListener('click', (e) => {
  if (e.target === document.getElementById('modal-overlay')) hideModal();
});

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------
async function apiPost(url, body, isFormData = false) {
  const options = { method: 'POST' };
  if (isFormData) {
    options.body = body;
  } else {
    options.headers = { 'Content-Type': 'application/json' };
    options.body = JSON.stringify(body);
  }
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `Request failed (${res.status})`);
  }
  return data;
}

// ---------------------------------------------------------------------------
// Import ONNX
// ---------------------------------------------------------------------------
document.getElementById('btn-import-onnx').addEventListener('click', () => {
  showModal(`
    <h3>Import ONNX Model</h3>
    <label>ONNX File</label>
    <input type="file" id="onnx-file" accept=".onnx">
    <label>Model Name (optional)</label>
    <input type="text" id="onnx-name" placeholder="e.g. yolox_s">
    <label>Family</label>
    <select id="onnx-family">
      <option value="YOLOX">YOLOX</option>
      <option value="NanoDet">NanoDet</option>
      <option value="PicoDet">PicoDet</option>
      <option value="FCOS">FCOS</option>
      <option value="CenterNet">CenterNet</option>
      <option value="PP-YOLOE">PP-YOLOE</option>
    </select>
    <label>Classes</label>
    <input type="number" id="onnx-classes" value="80" min="1">
    <label>Pretrained</label>
    <select id="onnx-pretrained">
      <option value="auto">Auto-detect</option>
      <option value="yes">Yes</option>
      <option value="no">No</option>
    </select>
    <button class="primary" id="onnx-submit">Import</button>
  `);
  document.getElementById('onnx-submit').addEventListener('click', async () => {
    const fileInput = document.getElementById('onnx-file');
    if (!fileInput.files.length) { showToast('Select an ONNX file', 'error'); return; }
    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    fd.append('name', document.getElementById('onnx-name').value);
    fd.append('family', document.getElementById('onnx-family').value);
    fd.append('classes', document.getElementById('onnx-classes').value);
    fd.append('pretrained', document.getElementById('onnx-pretrained').value);
    try {
      const scene = await apiPost('/api/import/onnx', fd, true);
      loadScene(scene);
      hideModal();
      showToast('ONNX model imported successfully');
    } catch (e) {
      showToast(e.message, 'error');
    }
  });
});

// ---------------------------------------------------------------------------
// Import YAML
// ---------------------------------------------------------------------------
document.getElementById('btn-import-yaml').addEventListener('click', () => {
  showModal(`
    <h3>Import YAML Architecture</h3>
    <label>YAML File</label>
    <input type="file" id="yaml-file" accept=".yaml,.yml">
    <button class="primary" id="yaml-submit">Import</button>
  `);
  document.getElementById('yaml-submit').addEventListener('click', async () => {
    const fileInput = document.getElementById('yaml-file');
    if (!fileInput.files.length) { showToast('Select a YAML file', 'error'); return; }
    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    try {
      const scene = await apiPost('/api/import/yaml', fd, true);
      loadScene(scene);
      hideModal();
      showToast('YAML architecture imported successfully');
    } catch (e) {
      showToast(e.message, 'error');
    }
  });
});

// ---------------------------------------------------------------------------
// Edit: Cut
// ---------------------------------------------------------------------------
document.getElementById('btn-cut').addEventListener('click', async () => {
  if (!currentScene) { showToast('Load a scene first', 'error'); return; }
  if (selectedBlocks.size === 0) { showToast('Select blocks to cut', 'error'); return; }
  try {
    const data = await apiPost('/api/edit/cut', { scene: currentScene, ids: [...selectedBlocks] });
    loadScene(data.scene);
    showToast(`Cut ${data.result.removed.length} block(s)`);
  } catch (e) {
    showToast(e.message, 'error');
  }
});

// ---------------------------------------------------------------------------
// Edit: Prune
// ---------------------------------------------------------------------------
document.getElementById('btn-prune').addEventListener('click', () => {
  if (!currentScene) { showToast('Load a scene first', 'error'); return; }
  if (selectedBlocks.size !== 1) { showToast('Select exactly one block to prune', 'error'); return; }
  const blockId = [...selectedBlocks][0];
  showModal(`
    <h3>Prune Block: ${blockId}</h3>
    <label>Parameter Name</label>
    <input type="text" id="prune-param" placeholder="e.g. out_channels">
    <label>New Value</label>
    <input type="number" id="prune-value" min="1">
    <button class="primary" id="prune-submit">Prune</button>
  `);
  document.getElementById('prune-submit').addEventListener('click', async () => {
    const param = document.getElementById('prune-param').value;
    const value = document.getElementById('prune-value').value;
    if (!param || !value) { showToast('Fill all fields', 'error'); return; }
    try {
      const data = await apiPost('/api/edit/prune', { scene: currentScene, id: blockId, param, value: parseInt(value) });
      loadScene(data.scene);
      hideModal();
      showToast('Block pruned');
    } catch (e) {
      showToast(e.message, 'error');
    }
  });
});

// ---------------------------------------------------------------------------
// Edit: Replace
// ---------------------------------------------------------------------------
document.getElementById('btn-replace').addEventListener('click', () => {
  if (!currentScene) { showToast('Load a scene first', 'error'); return; }
  if (selectedBlocks.size !== 1) { showToast('Select exactly one block to replace', 'error'); return; }
  const blockId = [...selectedBlocks][0];
  showModal(`
    <h3>Replace Block: ${blockId}</h3>
    <label>New Block Type</label>
    <input type="text" id="replace-type" placeholder="e.g. DWConvBlock">
    <button class="primary" id="replace-submit">Replace</button>
  `);
  document.getElementById('replace-submit').addEventListener('click', async () => {
    const newType = document.getElementById('replace-type').value;
    if (!newType) { showToast('Enter a block type', 'error'); return; }
    try {
      const data = await apiPost('/api/edit/replace', { scene: currentScene, id: blockId, type: newType });
      loadScene(data.scene);
      hideModal();
      showToast('Block replaced');
    } catch (e) {
      showToast(e.message, 'error');
    }
  });
});

// ---------------------------------------------------------------------------
// Edit: Fuse
// ---------------------------------------------------------------------------
document.getElementById('btn-fuse').addEventListener('click', async () => {
  if (!currentScene) { showToast('Load a scene first', 'error'); return; }
  try {
    const data = await apiPost('/api/edit/fuse', { scene: currentScene });
    loadScene(data.scene);
    showToast(`Fused — removed ${data.result.removed.length} block(s)`);
  } catch (e) {
    showToast(e.message, 'error');
  }
});

// ---------------------------------------------------------------------------
// Edit: Freeze / Unfreeze
// ---------------------------------------------------------------------------
document.getElementById('btn-freeze').addEventListener('click', async () => {
  if (!currentScene) { showToast('Load a scene first', 'error'); return; }
  if (selectedBlocks.size === 0) { showToast('Select blocks to freeze', 'error'); return; }
  try {
    const data = await apiPost('/api/edit/freeze', { scene: currentScene, ids: [...selectedBlocks], value: true });
    loadScene(data.scene);
    showToast(`Froze ${data.result.updated.length} block(s)`);
  } catch (e) {
    showToast(e.message, 'error');
  }
});

document.getElementById('btn-unfreeze').addEventListener('click', async () => {
  if (!currentScene) { showToast('Load a scene first', 'error'); return; }
  if (selectedBlocks.size === 0) { showToast('Select blocks to unfreeze', 'error'); return; }
  try {
    const data = await apiPost('/api/edit/freeze', { scene: currentScene, ids: [...selectedBlocks], value: false });
    loadScene(data.scene);
    showToast(`Unfroze ${data.result.updated.length} block(s)`);
  } catch (e) {
    showToast(e.message, 'error');
  }
});

// ---------------------------------------------------------------------------
// Export YAML
// ---------------------------------------------------------------------------
document.getElementById('btn-export-yaml').addEventListener('click', async () => {
  if (!currentScene) { showToast('Load a scene first', 'error'); return; }
  try {
    const data = await apiPost('/api/export/yaml', currentScene);
    showModal(`
      <h3>Exported YAML</h3>
      <textarea readonly>${data.yaml}</textarea>
    `);
  } catch (e) {
    showToast(e.message, 'error');
  }
});

// ---------------------------------------------------------------------------
// Export PyTorch
// ---------------------------------------------------------------------------
document.getElementById('btn-export-pytorch').addEventListener('click', () => {
  if (!currentScene) { showToast('Load a scene first', 'error'); return; }
  showModal(`
    <h3>Export PyTorch</h3>
    <label>Module Name</label>
    <input type="text" id="pt-module" value="GeneratedVoxelModel">
    <button class="primary" id="pt-submit">Export</button>
  `);
  document.getElementById('pt-submit').addEventListener('click', async () => {
    const moduleName = document.getElementById('pt-module').value || 'GeneratedVoxelModel';
    try {
      const data = await apiPost('/api/export/pytorch', { scene: currentScene, module_name: moduleName });
      showModal(`
        <h3>PyTorch Module</h3>
        <textarea readonly>${data.python}</textarea>
        <h3 style="margin-top:1rem;">Config JSON</h3>
        <textarea readonly style="min-height:100px">${data.config}</textarea>
      `);
    } catch (e) {
      showToast(e.message, 'error');
    }
  });
});

// ---------------------------------------------------------------------------
// Validate
// ---------------------------------------------------------------------------
document.getElementById('btn-validate').addEventListener('click', async () => {
  if (!currentScene) { showToast('Load a scene first', 'error'); return; }
  try {
    await apiPost('/api/validate', currentScene);
    showToast('✔ Scene is valid', 'success');
  } catch (e) {
    showToast(`✘ ${e.message}`, 'error');
  }
});

// ---------------------------------------------------------------------------
// Schedule (re-layout)
// ---------------------------------------------------------------------------
document.getElementById('btn-schedule').addEventListener('click', async () => {
  if (!currentScene) { showToast('Load a scene first', 'error'); return; }
  try {
    const scene = await apiPost('/api/schedule', currentScene);
    loadScene(scene);
    showToast('3D layout recalculated');
  } catch (e) {
    showToast(e.message, 'error');
  }
});

// ---------------------------------------------------------------------------
// Normalize (recolor based on block types)
// ---------------------------------------------------------------------------
document.getElementById('btn-normalize').addEventListener('click', async () => {
  if (!currentScene) { showToast('Load a scene first', 'error'); return; }
  try {
    const scene = await apiPost('/api/normalize', currentScene);
    loadScene(scene);
    showToast('Stage colors recomputed');
  } catch (e) {
    showToast(e.message, 'error');
  }
});

// ---------------------------------------------------------------------------
// Templates catalog (loaded lazily on first use)
// ---------------------------------------------------------------------------
let templatesCatalog = null;       // [{id, label, category, family, hyperparameters, blocks}]
let blockDefaultsCatalog = null;   // {blockType: {param: defaultValue}}

async function ensureTemplatesCatalog() {
  if (templatesCatalog !== null) return;
  const res = await fetch('/api/templates');
  const data = await res.json();
  templatesCatalog = data.templates || [];
  blockDefaultsCatalog = data.blockDefaults || {};
}

document.getElementById('btn-templates').addEventListener('click', async () => {
  if (!currentScene) { showToast('Load or create a scene first', 'error'); return; }
  try { await ensureTemplatesCatalog(); }
  catch (e) { showToast('Failed to load templates: ' + e.message, 'error'); return; }

  const categories = ['backbone', 'neck', 'head'];
  const grouped = { backbone: [], neck: [], head: [] };
  templatesCatalog.forEach(t => {
    if (grouped[t.category]) grouped[t.category].push(t);
  });

  let html = `<h3>🧩 Insert Template</h3>
    <p style="margin:0 0 0.5rem 0;opacity:0.8;font-size:0.85rem;">
      Anchor-free, ONNX-convertible presets. The template is appended after the
      currently selected block (or the last non-output block).
    </p>
    <label>Category</label>
    <select id="tpl-category">
      <option value="all">All</option>
      <option value="backbone">Backbone</option>
      <option value="neck">Neck</option>
      <option value="head">Head</option>
    </select>
    <label>Template</label>
    <select id="tpl-select" size="10" style="width:100%;font-family:monospace;"></select>
    <div id="tpl-description" style="margin:0.5rem 0;font-size:0.85rem;opacity:0.85;"></div>
    <div id="tpl-hyperparams" style="margin:0.5rem 0;font-size:0.8rem;font-family:monospace;background:#0f0f1a;padding:0.5rem;border-radius:4px;max-height:120px;overflow:auto;"></div>
    <button class="primary" id="tpl-submit">Insert</button>`;
  showModal(html);

  const categorySel = document.getElementById('tpl-category');
  const tplSel = document.getElementById('tpl-select');
  const descEl = document.getElementById('tpl-description');
  const hyperEl = document.getElementById('tpl-hyperparams');

  function refreshList() {
    const cat = categorySel.value;
    const items = templatesCatalog.filter(t => cat === 'all' || t.category === cat);
    tplSel.innerHTML = '';
    items.forEach(t => {
      const opt = document.createElement('option');
      opt.value = t.id;
      opt.textContent = `[${t.category}] ${t.family.padEnd(12)} — ${t.label}`;
      tplSel.appendChild(opt);
    });
    if (items.length > 0) {
      tplSel.value = items[0].id;
      refreshDescription();
    } else {
      descEl.textContent = '';
      hyperEl.textContent = '';
    }
  }

  function refreshDescription() {
    const tpl = templatesCatalog.find(t => t.id === tplSel.value);
    if (!tpl) return;
    descEl.textContent = tpl.description || '';
    hyperEl.textContent = JSON.stringify(tpl.hyperparameters || {}, null, 2);
  }

  categorySel.addEventListener('change', refreshList);
  tplSel.addEventListener('change', refreshDescription);
  refreshList();

  document.getElementById('tpl-submit').addEventListener('click', async () => {
    const templateId = tplSel.value;
    if (!templateId) { showToast('Pick a template', 'error'); return; }
    const anchorId = selectedBlocks.size === 1 ? [...selectedBlocks][0] : null;
    try {
      const data = await apiPost('/api/templates/insert', {
        scene: currentScene,
        template_id: templateId,
        anchor_id: anchorId,
      });
      loadScene(data.scene);
      hideModal();
      showToast(`Inserted ${data.result.inserted.length} block(s) from ${templateId}`);
    } catch (e) {
      showToast(e.message, 'error');
    }
  });
});


document.getElementById('btn-toggle-grid').addEventListener('click', () => {
  grid.visible = !grid.visible;
  showToast(grid.visible ? 'Grid shown' : 'Grid hidden');
});

// ---------------------------------------------------------------------------
// Tooltip on hover
// ---------------------------------------------------------------------------
canvas.addEventListener('mousemove', (e) => {
  if (isRotating || isPanning) {
    document.getElementById('block-tooltip').classList.add('hidden');
    return;
  }
  const rect = canvas.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const meshes = Object.values(blockMeshes);
  const intersects = raycaster.intersectObjects(meshes, true);
  const tooltip = document.getElementById('block-tooltip');
  if (intersects.length > 0) {
    let hitObj = intersects[0].object;
    while (hitObj && !hitObj.userData.block) {
      hitObj = hitObj.parent;
    }
    if (hitObj && hitObj.userData.block) {
      const b = hitObj.userData.block;
      const stage = (b.meta && b.meta.stage_id) || '?';
      const frozen = (b.meta && b.meta.frozen) ? ' ❄' : '';
      const anchorFreeRelevant = b.meta && b.meta.anchor_free_relevant !== false;
      const relevanceMarker = anchorFreeRelevant ? '' : ' ⚠';
      tooltip.textContent = `${b.id} [${b.type}] — ${stage}${frozen}${relevanceMarker}`;
      tooltip.style.left = (e.clientX - rect.left + 12) + 'px';
      tooltip.style.top = (e.clientY - rect.top + 12) + 'px';
      tooltip.classList.remove('hidden');
    } else {
      tooltip.classList.add('hidden');
    }
  } else {
    tooltip.classList.add('hidden');
  }
});
