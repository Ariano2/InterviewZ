/* ── Luminary — script.js ─────────────────────────────────────────────────── */

// ── Typing animation ─────────────────────────────────────────────────────────
const TYPING_ROLES = {{TYPING_ROLES_JSON}};
let roleIdx = 0, charIdx = 0, deleting = false;
const typingEl  = document.getElementById("typingText");
const aboutRole = document.getElementById("aboutRole");

function typeNext() {
  const role = TYPING_ROLES[roleIdx];
  if (!deleting) {
    typingEl.textContent = role.slice(0, ++charIdx);
    if (charIdx === role.length) {
      deleting = true;
      setTimeout(typeNext, 1800);
      return;
    }
  } else {
    typingEl.textContent = role.slice(0, --charIdx);
    if (charIdx === 0) {
      deleting = false;
      roleIdx = (roleIdx + 1) % TYPING_ROLES.length;
    }
  }
  setTimeout(typeNext, deleting ? 55 : 90);
}
if (typingEl) {
  typeNext();
  if (aboutRole) aboutRole.textContent = TYPING_ROLES[0];
}

// ── Navbar scroll state ───────────────────────────────────────────────────────
const navbar = document.getElementById("navbar");
window.addEventListener("scroll", () => {
  navbar.classList.toggle("scrolled", window.scrollY > 30);
  highlightNav();
}, { passive: true });

// ── Active nav link on scroll ─────────────────────────────────────────────────
function highlightNav() {
  const sections = document.querySelectorAll("section[id]");
  const scrollY  = window.scrollY + 90;
  sections.forEach(sec => {
    const top    = sec.offsetTop;
    const height = sec.offsetHeight;
    const link   = document.querySelector(`.nav-links a[href="#${sec.id}"]`);
    if (link) link.classList.toggle("active", scrollY >= top && scrollY < top + height);
  });
}

// ── Hamburger mobile menu ─────────────────────────────────────────────────────
const hamburger  = document.getElementById("hamburger");
const mobileMenu = document.getElementById("mobileMenu");
hamburger?.addEventListener("click", () => mobileMenu.classList.toggle("open"));
mobileMenu?.querySelectorAll("a").forEach(a => a.addEventListener("click", () => mobileMenu.classList.remove("open")));

// ── Intersection Observer — reveal animations ─────────────────────────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add("visible");
      observer.unobserve(e.target);
    }
  });
}, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });

document.querySelectorAll(".reveal, .project-card, .timeline-item").forEach(el => observer.observe(el));

// ── Skill chip stagger ────────────────────────────────────────────────────────
document.querySelectorAll(".skill-chip").forEach((chip, i) => {
  chip.style.animationDelay = `${i * 45}ms`;
});

// ── Animated stat counters ────────────────────────────────────────────────────
function animateCounter(el) {
  const target = parseInt(el.dataset.target, 10);
  const duration = 1200;
  const step = target / (duration / 16);
  let current = 0;
  const timer = setInterval(() => {
    current = Math.min(current + step, target);
    el.textContent = Math.floor(current);
    if (current >= target) clearInterval(timer);
  }, 16);
}

const statObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.querySelectorAll(".stat-num").forEach(animateCounter);
      statObserver.unobserve(e.target);
    }
  });
}, { threshold: 0.5 });
document.querySelectorAll(".stats-bar").forEach(el => statObserver.observe(el));

// ── Smooth scroll for anchor links ───────────────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener("click", e => {
    const target = document.querySelector(a.getAttribute("href"));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
});
