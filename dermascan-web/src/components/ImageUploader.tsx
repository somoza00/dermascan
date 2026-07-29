import { useCallback, useEffect, useState } from 'react';

interface ImageUploaderProps {
  onImageSelect: (file: File) => void;
  onRejected?: (reason: string) => void;
  disabled?: boolean;
}

const ACCEPTED_TYPES = ['image/jpeg', 'image/png'];
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10MB — mesmo limite aplicado pela API

export function ImageUploader({ onImageSelect, onRejected, disabled }: ImageUploaderProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  // Revoga a Object URL anterior sempre que uma nova é criada, e no unmount —
  // sem isso, cada imagem testada pelo usuário vaza memória no navegador.
  useEffect(() => {
    if (!preview) return;
    return () => URL.revokeObjectURL(preview);
  }, [preview]);

  const handleFile = useCallback((file: File) => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      onRejected?.('Formato não suportado. Envie uma imagem JPEG ou PNG.');
      return;
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      onRejected?.('Imagem muito grande. O limite é 10MB.');
      return;
    }
    setPreview(URL.createObjectURL(file));
    onImageSelect(file);
  }, [onImageSelect, onRejected]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const openFileDialog = useCallback(() => {
    if (!disabled) document.getElementById('file-input')?.click();
  }, [disabled]);

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      aria-label="Selecionar imagem da lesão de pele para análise. Arraste um arquivo ou pressione Enter para escolher."
      className="image-uploader-dropzone"
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={openFileDialog}
      onKeyDown={(e) => {
        if (disabled) return;
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          openFileDialog();
        }
      }}
      style={{
        border: `2px dashed ${dragOver ? '#3b82f6' : '#64748b'}`,
        borderRadius: 16,
        padding: 40,
        textAlign: 'center',
        cursor: disabled ? 'not-allowed' : 'pointer',
        background: dragOver ? '#1e293b' : '#0f172a',
        transition: 'all 0.2s',
        opacity: disabled ? 0.5 : 1,
      }}
    >
      <input
        id="file-input"
        type="file"
        accept="image/jpeg,image/png"
        onChange={handleChange}
        style={{ display: 'none' }}
        disabled={disabled}
        tabIndex={-1}
      />

      {preview ? (
        <img
          src={preview}
          alt="Pré-visualização da imagem da lesão selecionada para análise"
          style={{ maxHeight: 300, borderRadius: 12, maxWidth: '100%' }}
        />
      ) : (
        <div>
          <div style={{ fontSize: 48, marginBottom: 12 }}>📷</div>
          <p style={{ color: '#94a3b8', margin: 0 }}>
            Arraste uma imagem aqui ou clique para selecionar
          </p>
          <p style={{ color: '#64748b', fontSize: 14, marginTop: 8 }}>
            JPEG ou PNG • Máx 10MB
          </p>
        </div>
      )}
    </div>
  );
}
