function scrollToSection(id) {
  const section = document.getElementById(id);
  if (section) {
    section.scrollIntoView({ behavior: 'smooth' });
  }
}

function toggleMenu() {
  document.getElementById("sidebar").classList.toggle("open");
}
