import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { Plane, AlertTriangle, Eye, Globe, Radar, Gauge, Navigation, Activity } from 'lucide-react';

const GuardianEyeDashboard = () => {
  const [aircraftData, setAircraftData] = useState([]);
  const [tailNumbers, setTailNumbers] = useState([]);
  const [selectedAirline, setSelectedAirline] = useState('All Airlines');
  const [selectedTailNumber, setSelectedTailNumber] = useState('All Aircraft');
  const [selectedAircraftType, setSelectedAircraftType] = useState('All Types');
  const [currentTime, setCurrentTime] = useState(new Date());
  const [alertLevel, setAlertLevel] = useState('NORMAL');
  const globeRef = useRef();
  const rendererRef = useRef();
  const animationRef = useRef();

  // ------------------- FETCH REAL DATA -------------------
  useEffect(() => {
    fetch("https://your-app.streamlit.app/?data=flights")  // ✅ replace with your Streamlit Cloud URL
      .then(res => res.json())
      .then(data => {
        setAircraftData(data);
        setTailNumbers(['All Aircraft', ...data.map(d => d.tail_number)]);
      })
      .catch(err => console.error("Error fetching flight data:", err));
  }, []);

  // ------------------- FILTER -------------------
  const filteredAircraft = aircraftData.filter(ac => {
    if (selectedAirline !== 'All Airlines' && ac.airline !== selectedAirline) return false;
    if (selectedTailNumber !== 'All Aircraft' && ac.tail_number !== selectedTailNumber) return false;
    if (selectedAircraftType !== 'All Types' && ac.aircraft_type !== selectedAircraftType) return false;
    return true;
  });

  const availableTailNumbers = ['All Aircraft', ...aircraftData
    .filter(ac => {
      if (selectedAirline !== 'All Airlines' && ac.airline !== selectedAirline) return false;
      if (selectedAircraftType !== 'All Types' && ac.aircraft_type !== selectedAircraftType) return false;
      return true;
    })
    .map(ac => ac.tail_number)];

  // ------------------- 3D Globe -------------------
  useEffect(() => {
    if (!globeRef.current) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, 400 / 300, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(400, 300);
    renderer.setClearColor(0x000000, 0);
    globeRef.current.appendChild(renderer.domElement);

    // Earth
    const earth = new THREE.Mesh(
      new THREE.SphereGeometry(2, 32, 32),
      new THREE.MeshBasicMaterial({ color: 0x2563EB, wireframe: true, transparent: true, opacity: 0.3 })
    );
    scene.add(earth);

    // Aircraft markers
    filteredAircraft.forEach(ac => {
      const phi = (90 - ac.current_lat) * (Math.PI / 180);
      const theta = (ac.current_lng + 180) * (Math.PI / 180);
      const x = 2.1 * Math.sin(phi) * Math.cos(theta);
      const y = 2.1 * Math.cos(phi);
      const z = 2.1 * Math.sin(phi) * Math.sin(theta);

      const color =
        ac.risk_level === 'CRITICAL' ? 0xFF0000 :
        ac.risk_level === 'HIGH' ? 0xFFAA00 :
        ac.risk_level === 'MEDIUM' ? 0x0088FF : 0x00FF00;

      const aircraftMesh = new THREE.Mesh(
        new THREE.BoxGeometry(0.1, 0.02, 0.1),
        new THREE.MeshBasicMaterial({ color })
      );
      aircraftMesh.position.set(x, y, z);
      scene.add(aircraftMesh);
    });

    camera.position.z = 5;

    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);
      earth.rotation.y += 0.005;
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (globeRef.current && renderer.domElement) globeRef.current.removeChild(renderer.domElement);
    };
  }, [filteredAircraft]);

  // ------------------- Alert System -------------------
  useEffect(() => {
    const criticalCount = filteredAircraft.filter(a => a.risk_level === 'CRITICAL').length;
    const highCount = filteredAircraft.filter(a => a.risk_level === 'HIGH').length;
    if (criticalCount > 0) setAlertLevel('CRITICAL');
    else if (highCount > 2) setAlertLevel('HIGH');
    else if (highCount > 0) setAlertLevel('ELEVATED');
    else setAlertLevel('NORMAL');
  }, [filteredAircraft]);

  // ------------------- Selected Aircraft -------------------
  const selectedAircraftData = selectedTailNumber === 'All Aircraft'
    ? null
    : aircraftData.find(ac => ac.tail_number === selectedTailNumber);

  // ------------------- UI -------------------
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-black text-white">
      {/* Header */}
      <div className="bg-black bg-opacity-50 border-b border-blue-500/30 p-4 flex justify-between">
        <div className="flex items-center space-x-3">
          <Eye className="w-8 h-8 text-blue-400" />
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">GUARDIAN EYE</h1>
        </div>
        <div className={`px-4 py-2 rounded-lg font-bold ${
          alertLevel === 'CRITICAL' ? 'bg-red-600 animate-pulse' :
          alertLevel === 'HIGH' ? 'bg-orange-600' :
          alertLevel === 'ELEVATED' ? 'bg-yellow-600 text-black' : 'bg-green-600'
        }`}>ALERT: {alertLevel}</div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-4 p-4 h-screen">
        {/* Filters */}
        <div className="col-span-3">
          <h3 className="text-lg font-bold mb-2">Aircraft Selection</h3>
          <select value={selectedAirline} onChange={e => setSelectedAirline(e.target.value)} className="w-full mb-2 bg-gray-800 p-2">
            {['All Airlines', ...new Set(aircraftData.map(a => a.airline))].map(a => <option key={a}>{a}</option>)}
          </select>
          <select value={selectedAircraftType} onChange={e => setSelectedAircraftType(e.target.value)} className="w-full mb-2 bg-gray-800 p-2">
            {['All Types', ...new Set(aircraftData.map(a => a.aircraft_type))].map(t => <option key={t}>{t}</option>)}
          </select>
          <select value={selectedTailNumber} onChange={e => setSelectedTailNumber(e.target.value)} className="w-full bg-gray-800 p-2">
            {availableTailNumbers.map(t => <option key={t}>{t}</option>)}
          </select>
        </div>

        {/* Globe */}
        <div className="col-span-6">
          <h3 className="text-lg font-bold mb-2">🌐 Global Aircraft Tracking</h3>
          <div ref={globeRef}></div>
        </div>

        {/* Details */}
        <div className="col-span-3">
          {selectedAircraftData ? (
            <div className="bg-black p-4 rounded">
              <h3 className="font-bold mb-2">{selectedAircraftData.tail_number}</h3>
              <p>{selectedAircraftData.airline} — {selectedAircraftData.aircraft_type}</p>
              <p>Status: {selectedAircraftData.status}</p>
              <p>Risk: {selectedAircraftData.risk_level} ({(selectedAircraftData.incident_probability * 100).toFixed(1)}%)</p>
              <p>Delay: {selectedAircraftData.delay_minutes} min</p>
            </div>
          ) : <p>Select an aircraft for details</p>}
        </div>
      </div>
    </div>
  );
};

export default GuardianEyeDashboard;
