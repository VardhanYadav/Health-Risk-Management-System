// Toggle FAQ answers
document.querySelectorAll('.faq-question').forEach(question => {
    question.addEventListener('click', () => {
        // Toggle active class for styling
        question.classList.toggle('active');
        
        // Toggle display of answer
        const answer = question.nextElementSibling;
        answer.classList.toggle('show');
    });
});

// Filter FAQs by category
const categorySelect = document.getElementById('faq-category');
const faqSections = {
    'all': ['general-info', 'data-tech', 'features-usage', 'health-privacy', 'technical'],
    'general': ['general-info'],
    'data': ['data-tech'],
    'features': ['features-usage'],
    'health': ['health-privacy'],
    'technical': ['technical']
};

categorySelect.addEventListener('change', () => {
    const category = categorySelect.value;
    
    // Hide all sections first
    document.querySelectorAll('.faq-section').forEach(section => {
        section.style.display = 'none';
    });
    
    // Show only sections for selected category
    faqSections[category].forEach(sectionId => {
        document.getElementById(sectionId).style.display = 'block';
    });
});