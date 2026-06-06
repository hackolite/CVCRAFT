/**
 * CVCRAFT Web UI — Three.js 3D Scene Viewer + API Client
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let currentScene = null;
let selectedBlocks = new Set();
let blockMeshes = {};

// ---------------------------------------------------------------------------
// Three.js Setup
// ---------------------------------------------------------------------------
const canvas = document.getElementById('viewport');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);

const scene3d = new THREE.Scene();
scene3d.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
camera.position.set(30, 20, 30);
camera.lookAt(0, 0, 0);

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
scene3d.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(20, 30, 20);
scene3d.add(dirLight);

// Grid helper
const grid = new THREE.GridHelper(60, 60, 0x333355, 0x222244);
scene3d.add(grid);

// Stage colors
const STAGE_COLORS = {
  input: 0x9CA3AF,
  backbone: 0x2563EB,
  neck: 0xD97706,
  head: 0xDC2626,
  output: 0x059669,
};

const FROZEN_OPACITY = 0.4;

// ---------------------------------------------------------------------------
// Orbit Controls (minimal implementation)
// ---------------------------------------------------------------------------
let isDragging = false;
let dragMoved = false;
let prevMouse = { x: 0, y: 0 };
let spherical = { theta: Math.PI / 4, phi: Math.PI / 4, radius: 50 };

function updateCameraFromSpherical() {
  camera.position.x = spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta);
  camera.position.y = spherical.radius * Math.cos(spherical.phi);
  camera.position.z = spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta);
  camera.lookAt(0, 0, 0);
}

canvas.addEventListener('mousedown', (e) => {
  isDragging = true;
  dragMoved = false;
  prevMouse = { x: e.clientX, y: e.clientY };
});

canvas.addEventListener('mousemove', (e) => {
  if (!isDragging) return;
  dragMoved = true;
  const dx = e.clientX - prevMouse.x;
  const dy = e.clientY - prevMouse.y;
  spherical.theta -= dx * 0.005;
  spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi + dy * 0.005));
  updateCameraFromSpherical();
  prevMouse = { x: e.clientX, y: e.clientY };
});

canvas.addEventListener('mouseup', () => { isDragging = false; });
canvas.addEventListener('mouseleave', () => { isDragging = false; });

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  spherical.radius = Math.max(5, Math.min(150, spherical.radius + e.deltaY * 0.05));
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
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

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

  blocks.forEach(block => {
    const stageId = (block.meta && block.meta.stage_id) || 'backbone';
    const frozen = block.meta && block.meta.frozen;
    const color = STAGE_COLORS[stageId] || 0x2563EB;

    const geometry = new THREE.BoxGeometry(1.4, 1.4, 1.4);
    const material = new THREE.MeshPhongMaterial({
      color,
      transparent: frozen,
      opacity: frozen ? FROZEN_OPACITY : 1.0,
    });
    const mesh = new THREE.Mesh(geometry, material);

    const pos = block.position || { x: 0, y: 0, z: 0 };
    mesh.position.set(pos.x, pos.y, pos.z);
    mesh.userData = { blockId: block.id, block };
    scene3d.add(mesh);
    blockMeshes[block.id] = mesh;
  });

  // Draw edges as lines
  const blockById = {};
  blocks.forEach(b => { blockById[b.id] = b; });

  edges.forEach(edge => {
    const fromBlock = blockById[edge.from];
    const toBlock = blockById[edge.to];
    if (!fromBlock || !toBlock) return;
    const from = fromBlock.position || { x: 0, y: 0, z: 0 };
    const to = toBlock.position || { x: 0, y: 0, z: 0 };

    const points = [
      new THREE.Vector3(from.x, from.y, from.z),
      new THREE.Vector3(to.x, to.y, to.z),
    ];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: 0x555588, linewidth: 1 });
    const line = new THREE.Line(geometry, material);
    line.userData = { isEdge: true };
    scene3d.add(line);
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
  const intersects = raycaster.intersectObjects(meshes);

  if (intersects.length > 0) {
    const hit = intersects[0].object;
    const blockId = hit.userData.blockId;

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
  selectedBlocks.forEach(id => {
    const mesh = blockMeshes[id];
    if (!mesh) return;
    const b = mesh.userData.block;
    lines.push(`ID: ${b.id}\nType: ${b.type}\nStage: ${b.meta?.stage_id || '?'}\nFrozen: ${b.meta?.frozen || false}\n`);
  });
  info.textContent = lines.join('\n---\n');
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

function loadScene(sceneData) {
  currentScene = sceneData;
  selectedBlocks.clear();
  renderScene(sceneData);
  updateMetrics();
  updateSceneInfo();
  updateSelectionInfo();
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
// Tooltip on hover
// ---------------------------------------------------------------------------
canvas.addEventListener('mousemove', (e) => {
  if (isDragging) {
    document.getElementById('block-tooltip').classList.add('hidden');
    return;
  }
  const rect = canvas.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const meshes = Object.values(blockMeshes);
  const intersects = raycaster.intersectObjects(meshes);
  const tooltip = document.getElementById('block-tooltip');
  if (intersects.length > 0) {
    const hit = intersects[0].object;
    const b = hit.userData.block;
    const stage = (b.meta && b.meta.stage_id) || '?';
    const frozen = (b.meta && b.meta.frozen) ? ' ❄' : '';
    tooltip.textContent = `${b.id} [${b.type}] — ${stage}${frozen}`;
    tooltip.style.left = (e.clientX - rect.left + 12) + 'px';
    tooltip.style.top = (e.clientY - rect.top + 12) + 'px';
    tooltip.classList.remove('hidden');
  } else {
    tooltip.classList.add('hidden');
  }
});
