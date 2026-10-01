import React from 'react';

/**
 * Reusable section card with consistent header/icon/title pattern.
 *
 * Props:
 *   icon - emoji or React node for the icon
 *   title - section title string
 *   subtitle - optional subtitle
 *   children - card body content
 *   className - additional className
 *   fullWidth - if true, adds full-width class for the grid
 */
export default function SummaryCard({
  icon,
  title,
  subtitle,
  children,
  className = '',
  fullWidth = false,
}) {
  return (
    <div className={`card ${fullWidth ? 'full-width' : ''} ${className}`}>
      <div className="card-header">
        {icon && (
          <div className="card-icon" aria-hidden="true">
            {icon}
          </div>
        )}
        <div>
          <div className="card-title">{title}</div>
          {subtitle && <div className="card-subtitle">{subtitle}</div>}
        </div>
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}
