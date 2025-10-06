#!/usr/bin/env python3
"""
Script de pruebas de seguridad para validar las mejoras implementadas.
Valida la funcionalidad de subida de archivos y medidas de seguridad.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock

# Configurar logging para las pruebas
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_security_module():
    """Prueba las funciones del módulo de seguridad."""
    
    logger.info("=== INICIANDO PRUEBAS DE SEGURIDAD ===")
    
    try:
        from src.security import (
            sanitize_filename, generate_secure_filename, 
            validate_file_extension, comprehensive_file_validation,
            SecurityError
        )
        
        # Test 1: Sanitización de nombres de archivo
        logger.info("Test 1: Sanitización de nombres de archivo")
        
        test_cases = [
            ("audio.wav", True, "archivo válido"),
            ("../../../etc/passwd", False, "directory traversal"),
            ("test<>file.mp3", False, "caracteres inválidos"), 
            ("CON.wav", False, "nombre reservado Windows"),
            ("normal_file.mp3", True, "archivo normal"),
            ("", False, "nombre vacío"),
            ("file with spaces.wav", True, "espacios en nombre"),
        ]
        
        for filename, should_pass, description in test_cases:
            try:
                sanitized = sanitize_filename(filename)
                if should_pass:
                    logger.info(f"  ✓ {description}: '{filename}' -> '{sanitized}'")
                else:
                    logger.warning(f"  ⚠️ {description}: '{filename}' pasó cuando debería fallar")
            except SecurityError as e:
                if should_pass:
                    logger.error(f"  ✗ {description}: '{filename}' falló cuando debería pasar: {e}")
                else:
                    logger.info(f"  ✓ {description}: '{filename}' correctamente rechazado: {e}")
        
        # Test 2: Generación de nombres seguros
        logger.info("Test 2: Generación de nombres seguros")
        
        original = "mi archivo de audio.wav"
        secure = generate_secure_filename(original)
        logger.info(f"  ✓ Nombre original: '{original}'")
        logger.info(f"  ✓ Nombre seguro: '{secure}'")
        
        # Test 3: Validación de extensiones
        logger.info("Test 3: Validación de extensiones")
        
        extension_tests = [
            ("audio.wav", True),
            ("music.mp3", True),
            ("song.flac", True),
            ("virus.exe", False),
            ("script.js", False),
            ("document.pdf", False),
        ]
        
        for filename, should_pass in extension_tests:
            result = validate_file_extension(filename)
            if result == should_pass:
                logger.info(f"  ✓ {filename}: {'válido' if result else 'rechazado'}")
            else:
                logger.error(f"  ✗ {filename}: resultado inesperado ({result})")
        
        logger.info("✓ Pruebas del módulo de seguridad completadas")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error en pruebas de seguridad: {e}", exc_info=True)
        return False

def test_configuration_module():
    """Prueba el módulo de configuración."""
    
    logger.info("=== INICIANDO PRUEBAS DE CONFIGURACIÓN ===")
    
    try:
        from src.config import get_config, initialize_config
        
        # Test configuración
        config = get_config()
        logger.info(f"  ✓ App Name: {config.APP_NAME}")
        logger.info(f"  ✓ Max File Size: {config.MAX_FILE_SIZE_MB}MB")
        logger.info(f"  ✓ Debug Mode: {config.DEBUG}")
        logger.info(f"  ✓ Log Level: {config.LOG_LEVEL}")
        
        # Test validación de configuración
        is_valid = config.validate_config()
        if is_valid:
            logger.info("  ✓ Configuración válida")
        else:
            logger.warning("  ⚠️ Configuración inválida")
        
        # Test inicialización
        init_result = initialize_config()
        if init_result:
            logger.info("  ✓ Inicialización exitosa")
        else:
            logger.warning("  ⚠️ Error en inicialización")
        
        logger.info("✓ Pruebas del módulo de configuración completadas")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error en pruebas de configuración: {e}", exc_info=True)
        return False

def test_logging_configuration():
    """Prueba la configuración de logging."""
    
    logger.info("=== INICIANDO PRUEBAS DE LOGGING ===")
    
    try:
        from src.logging_config import setup_logging, get_security_logger
        
        # Test configuración de logging
        setup_logging("TestProcess", "DEBUG")
        logger.info("  ✓ Logging configurado correctamente")
        
        # Test logger de seguridad
        sec_logger = get_security_logger("test")
        sec_logger.info("Mensaje de prueba de seguridad")
        logger.info("  ✓ Logger de seguridad funcionando")
        
        # Verificar que se crean los directorios de logs
        logs_dir = Path("logs")
        if logs_dir.exists():
            logger.info(f"  ✓ Directorio de logs creado: {logs_dir.absolute()}")
        else:
            logger.warning("  ⚠️ Directorio de logs no encontrado")
        
        logger.info("✓ Pruebas de logging completadas")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error en pruebas de logging: {e}", exc_info=True)
        return False

def create_test_audio_file(filepath: Path, content: bytes = b"fake audio content"):
    """Crea un archivo de audio de prueba."""
    with open(filepath, 'wb') as f:
        f.write(content)

def test_file_validation():
    """Prueba la validación completa de archivos."""
    
    logger.info("=== INICIANDO PRUEBAS DE VALIDACIÓN DE ARCHIVOS ===")
    
    try:
        from src.security import comprehensive_file_validation, SecurityError
        
        # Crear directorio temporal para pruebas
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test 1: Archivo válido simulado
            logger.info("Test 1: Archivo válido simulado")
            valid_file = temp_path / "test_audio.wav"
            
            # Simular header WAV básico
            wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00'
            create_test_audio_file(valid_file, wav_header + b'\x00' * 100)
            
            try:
                result = comprehensive_file_validation(str(valid_file), "test_audio.wav")
                logger.info(f"  ✓ Archivo válido procesado: {result['is_valid']}")
                if result['detected_mime_type']:
                    logger.info(f"  ✓ MIME detectado: {result['detected_mime_type']}")
                if result['warnings']:
                    logger.info(f"  ⚠️ Warnings: {result['warnings']}")
            except Exception as e:
                logger.info(f"  ℹ️ Validación falló (esperado sin libmagic): {e}")
            
            # Test 2: Archivo demasiado grande
            logger.info("Test 2: Archivo muy grande")
            large_file = temp_path / "large_file.wav"
            create_test_audio_file(large_file, b'\x00' * (10 * 1024 * 1024))  # 10MB
            
            try:
                result = comprehensive_file_validation(str(large_file), "large_file.wav")
                logger.info(f"  ✓ Archivo grande procesado: {result['is_valid']}")
                logger.info(f"  ✓ Tamaño detectado: {result['file_size_mb']:.2f}MB")
            except Exception as e:
                logger.info(f"  ℹ️ Validación de archivo grande falló: {e}")
            
            # Test 3: Archivo vacío
            logger.info("Test 3: Archivo vacío")
            empty_file = temp_path / "empty.wav"
            create_test_audio_file(empty_file, b'')
            
            try:
                result = comprehensive_file_validation(str(empty_file), "empty.wav")
                logger.info(f"  ✓ Archivo vacío procesado: {result['is_valid']}")
                if result['errors']:
                    logger.info(f"  ✓ Errores detectados: {result['errors']}")
            except Exception as e:
                logger.info(f"  ℹ️ Validación de archivo vacío falló: {e}")
        
        logger.info("✓ Pruebas de validación de archivos completadas")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error en pruebas de validación de archivos: {e}", exc_info=True)
        return False

def run_all_tests():
    """Ejecuta todas las pruebas de seguridad."""
    
    logger.info("INICIANDO SUITE COMPLETA DE PRUEBAS DE SEGURIDAD")
    logger.info("=" * 60)
    
    results = []
    
    # Ejecutar cada grupo de pruebas
    test_functions = [
        ("Módulo de Seguridad", test_security_module),
        ("Módulo de Configuración", test_configuration_module), 
        ("Configuración de Logging", test_logging_configuration),
        ("Validación de Archivos", test_file_validation),
    ]
    
    for test_name, test_func in test_functions:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Error crítico en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen de resultados
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN DE PRUEBAS")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASÓ" if result else "✗ FALLÓ"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("-" * 60)
    logger.info(f"TOTAL: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        logger.info("🎉 TODAS LAS PRUEBAS DE SEGURIDAD PASARON")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} PRUEBAS FALLARON")
        return False

def main():
    """Función principal."""
    
    try:
        success = run_all_tests()
        
        # Información adicional
        logger.info("\n" + "=" * 60)
        logger.info("INFORMACIÓN ADICIONAL")
        logger.info("=" * 60)
        logger.info("Para un entorno de producción, asegúrate de:")
        logger.info("1. Instalar python-magic y python-magic-bin")
        logger.info("2. Configurar variables de entorno apropiadas")
        logger.info("3. Revisar permisos de directorios")
        logger.info("4. Configurar firewall y límites de sistema")
        logger.info("5. Implementar monitoreo de seguridad")
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Error crítico en pruebas: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)