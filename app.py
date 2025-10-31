<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Admin - Aviator Pronosticador IA</title>
<style>
    body { background:#000; color:#0ff; font-family: Arial, sans-serif; padding:20px; }
    .wrap { max-width:900px; margin:0 auto; }
    .panel { background:#111; padding:16px; border-radius:10px; box-shadow:0 0 12px cyan; }
    input, select { padding:8px; margin:6px 0; border-radius:6px; border:1px solid #0ff; background:#000; color:#0ff; width:100%; }
    button { padding:8px 12px; background:#0ff; color:#000; border:none; border-radius:6px; cursor:pointer; }
    table { width:100%; border-collapse:collapse; margin-top:12px; }
    th, td { padding:8px; text-align:left; border-bottom:1px solid #222; }
    .danger { background:#f44;color:#fff;border:0;padding:6px 10px;border-radius:6px;cursor:pointer; }
    .small { font-size:0.9em; color:#9ff; }
    .topbar { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }
    a.logout { color:#0ff; text-decoration:none; border:1px solid #0ff; padding:6px 10px; border-radius:6px; }
</style>
</head>
<body>
<div class="wrap">
    <div class="topbar">
        <h2>🛠️ Panel Admin</h2>
        <div>
            <span class="small">Admin: {{ admin }}</span>
            &nbsp; <a class="logout" href="{{ url_for('logout') }}">Cerrar sesión</a>
        </div>
    </div>

    <div class="panel">
        <h3>Crear usuario</h3>
        <form id="formCrear" method="POST" action="{{ url_for('admin_panel') }}">
            <label>Correo</label>
            <input name="email" type="email" placeholder="usuario@ejemplo.com" required />
            <label>Contraseña</label>
            <input name="password" type="text" placeholder="contraseña" required />
            <label>Días de validez (1-30)</label>
            <input name="dias" type="number" min="1" max="3650" value="1" required />
            <button type="submit">Crear usuario</button>
        </form>
    </div>

    <div class="panel" style="margin-top:12px;">
        <h3>Usuarios existentes</h3>
        <table>
            <thead><tr><th>Email</th><th>Creado</th><th>Expira</th><th>Admin</th><th>Acciones</th></tr></thead>
            <tbody id="users_table">
            {% for u in users %}
                <tr>
                    <td>{{ u.email }}</td>
                    <td>{{ u.created }}</td>
                    <td>{{ u.expires }}</td>
                    <td>{{ 'Sí' if u.is_admin else 'No' }}</td>
                    <td>
                        {% if not u.is_admin %}
                        <button class="danger" onclick="eliminarUsuario({{ u.id }}, '{{ u.email }}')">Eliminar</button>
                        <button onclick="extenderUsuario({{ u.id }}, '{{ u.email }}')">Extender</button>
                        {% else %}
                        <span class="small">Cuenta admin</span>
                        {% endif %}
                    </td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </div>

    <div style="margin-top:12px;">
        <a href="{{ url_for('index') }}" style="color:#0ff">Ir a la app</a>
    </div>
</div>

<script>
function eliminarUsuario(id, email){
    if(!confirm("¿Eliminar usuario " + email + "?")) return;
    window.location.href = "/eliminar_usuario/" + id;
}

function extenderUsuario(id, email){
    const dias = prompt("¿Cuántos días deseas extender para " + email + "?", "7");
    if(!dias) return;
    const form = document.createElement("form");
    form.method = "POST";
    form.action = "/extender_usuario/" + id;
    const input = document.createElement("input");
    input.type = "hidden";
    input.name = "dias";
    input.value = dias;
    form.appendChild(input);
    document.body.appendChild(form);
    form.submit();
}
</script>
</body>
</html>
