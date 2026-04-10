/* ── Noir — script.js ─────────────────────────────────────────────────────── */

// ── Typing animation ─────────────────────────────────────────────────────────
const TYPING_ROLES = {{TYPING_ROLES_JSON}};
let roleIdx = 0, charIdx = 0, deleting = false;
const typingEl = document.getElementById("typingText");

function typeNext() {
  const role = TYPING_ROLES[roleIdx];
  if (!deleting) {
    typingEl.textContent = role.slice(0, ++charIdx);
    if (charIdx === role.length) { deleting = true; setTimeout(typeNext, 2000); return; }
  } else {
    typingEl.textContent = role.slice(0, --charIdx);
    if (charIdx === 0) { deleting = false; roleIdx = (roleIdx + 1) % TYPING_ROLES.length; }
  }
  setTimeout(typeNext, deleting ? 45 : 80);
}
if (typingEl) typeNext();

// ── Terminal about.json ───────────────────────────────────────────────────────
const terminalBody = document.getElementById("terminalBody");
if (terminalBody) {
  const name   = document.title.replace(" — Portfolio", "");
  const role   = TYPING_ROLES[0] || "Developer";
  const lines  = [
    `<span class="t-brace">{</span>`,
    `  <span class="t-key">"name"</span>: <span class="t-str">"${name}"</span>,`,
    `  <span class="t-key">"role"</span>: <span class="t-str">"${role}"</span>,`,
    `  <span class="t-key">"open_to_work"</span>: <span class="t-num">true</span>,`,
    `  <span class="t-key">"coffee_driven"</span>: <span class="t-num">true</span>,`,
    `  <span class="t-key">"bugs_written"</span>: <span class="t-num">404</span>`,
    `<span class="t-brace">}</span>`,
  ];
  let i = 0;
  const interval = setInterval(() => {
    if (i < lines.length) {
      const p = document.createElement("p");
      p.innerHTML = lines[i++];
      terminalBody.appendChild(p);
    } else {
      clearInterval(interval);
    }
  }, 200);
}

// ── Navbar scroll ─────────────────────────────────────────────────────────────
const navbar = document.getElementById("navbar");
window.addEventListener("scroll", () => {
  navbar.classList.toggle("scrolled", window.scrollY > 30);
  highlightNav();
}, { passive: true });

function highlightNav() {
  const sections = document.querySelectorAll("section[id]");
  const scrollY  = window.scrollY + 90;
  sections.forEach(sec => {
    const link = document.querySelector(`.nav-links a[href="#${sec.id}"]`);
    if (link) link.classList.toggle("active", scrollY >= sec.offsetTop && scrollY < sec.offsetTop + sec.offsetHeight);
  });
}

// ── Hamburger ─────────────────────────────────────────────────────────────────
const hamburger  = document.getElementById("hamburger");
const mobileMenu = document.getElementById("mobileMenu");
hamburger?.addEventListener("click", () => mobileMenu.classList.toggle("open"));
mobileMenu?.querySelectorAll("a").forEach(a => a.addEventListener("click", () => mobileMenu.classList.remove("open")));

// ── Intersection Observer ─────────────────────────────────────────────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) { e.target.classList.add("visible"); observer.unobserve(e.target); }
  });
}, { threshold: 0.1, rootMargin: "0px 0px -40px 0px" });

document.querySelectorAll(".reveal, .project-card, .timeline-item").forEach(el => observer.observe(el));

// ── Skill chip stagger ────────────────────────────────────────────────────────
document.querySelectorAll(".skill-chip").forEach((chip, i) => {
  chip.style.animationDelay = `${i * 40}ms`;
});

// ── Stat counters ─────────────────────────────────────────────────────────────
function animateCounter(el) {
  const target = parseInt(el.dataset.target, 10);
  const step   = target / (1200 / 16);
  let current  = 0;
  const t = setInterval(() => {
    current = Math.min(current + step, target);
    el.textContent = Math.floor(current);
    if (current >= target) clearInterval(t);
  }, 16);
}
const statObs = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) { e.target.querySelectorAll(".stat-num").forEach(animateCounter); statObs.unobserve(e.target); }
  });
}, { threshold: 0.5 });
document.querySelectorAll(".stats-bar").forEach(el => statObs.observe(el));

// ── Smooth scroll ─────────────────────────────────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener("click", e => {
    const target = document.querySelector(a.getAttribute("href"));
    if (target) { e.preventDefault(); target.scrollIntoView({ behavior: "smooth", block: "start" }); }
  });
});
